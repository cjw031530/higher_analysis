"""Read the monthly CSV and encode features without validation leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm


ALL_MONTHS = (
    202306, 202307, 202308, 202309, 202310, 202311,
    202312, 202401, 202402, 202403, 202404, 202405,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    train_months: tuple[int, ...]
    valid_months: tuple[int, ...]
    month_shares: dict[int, float] | None


def experiment_from_config(name: str, cfg: DictConfig) -> Experiment:
    protocol = cfg.protocols[name]
    shares = protocol.validation_month_shares
    result = Experiment(
        name=name,
        train_months=tuple(int(month) for month in protocol.train_months),
        valid_months=tuple(int(month) for month in protocol.validation_months),
        month_shares=None if shares is None else {int(k): float(v) for k, v in shares.items()},
    )
    if not result.train_months or not result.valid_months:
        raise ValueError("Training and validation months must both be nonempty.")
    if len(set(result.train_months)) != len(result.train_months) or len(set(result.valid_months)) != len(result.valid_months):
        raise ValueError("Months must be unique within each split.")
    if set(result.train_months) & set(result.valid_months):
        raise ValueError("Training and validation months overlap.")
    if max(result.train_months) >= min(result.valid_months):
        raise ValueError("Training months must precede validation months.")
    if result.month_shares is not None and (
        set(result.month_shares) != set(result.valid_months)
        or any(share <= 0 for share in result.month_shares.values())
        or not np.isclose(sum(result.month_shares.values()), 1.0)
    ):
        raise ValueError("Validation month shares must be positive, complete, and total one.")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path, chunk_rows: int, require_all_months: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if chunk_rows < 1:
        raise ValueError("data.chunk_rows must be positive.")
    chunks: list[pd.DataFrame] = []
    with tqdm(desc="Reading CSV", unit=" rows", position=0) as progress:
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            chunks.append(chunk)
            progress.update(len(chunk))
    if not chunks:
        raise ValueError("The source CSV is empty.")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    if not frame.columns.is_unique or not {"LNMON", "TARGET"}.issubset(frame.columns):
        raise ValueError("The CSV needs unique columns including LNMON and TARGET.")
    if frame[["LNMON", "TARGET"]].isna().any(axis=None):
        raise ValueError("LNMON and TARGET cannot be missing.")
    frame["LNMON"] = pd.to_numeric(frame["LNMON"], errors="raise")
    frame["TARGET"] = pd.to_numeric(frame["TARGET"], errors="raise")
    if not frame["LNMON"].isin(ALL_MONTHS).all():
        raise ValueError("LNMON contains a month outside 202306 through 202405.")
    if not frame["TARGET"].isin((0, 1)).all():
        raise ValueError("TARGET must contain only 0 and 1.")
    observed = set(frame["LNMON"].unique())
    if require_all_months and observed != set(ALL_MONTHS):
        raise ValueError(f"The CSV is missing required months: {sorted(set(ALL_MONTHS) - observed)}")
    frame["LNMON"] = frame["LNMON"].astype("int32")
    frame["TARGET"] = frame["TARGET"].astype("int8")
    if len(frame.columns) < 3:
        raise ValueError("The CSV has no predictor columns.")
    for name in tqdm(frame.select_dtypes(include="number").columns, desc="Compacting numeric columns", unit=" columns"):
        series = frame[name]
        if pd.api.types.is_integer_dtype(series):
            frame[name] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            values = series.dropna().to_numpy(dtype="float64", copy=False)
            if not len(values):
                frame[name] = series.astype("float32")
            elif (
                np.isfinite(values).all()
                and values.min() >= -(2**31)
                and values.max() <= 2**31 - 1
                and np.equal(values, np.trunc(values)).all()
            ):
                frame[name] = series.astype("Int32")
    return frame


def feature_schema(frame: pd.DataFrame, train_mask: np.ndarray) -> dict:
    columns = [name for name in frame.columns if name != "TARGET"]
    categorical = [
        name for name in columns
        if pd.api.types.is_object_dtype(frame[name])
        or pd.api.types.is_string_dtype(frame[name])
        or isinstance(frame[name].dtype, pd.CategoricalDtype)
    ]
    numeric = [name for name in columns if name not in categorical]
    levels = {
        name: sorted(frame.loc[train_mask, name].astype("string").dropna().unique().tolist())
        for name in categorical
    }
    encoded = list(numeric)
    for name in categorical:
        encoded.extend(f"{name}={level}" for level in levels[name])
        encoded.append(f"{name}=<MISSING_OR_UNKNOWN>")
    return {
        "feature_columns": columns,
        "numeric_columns": numeric,
        "categorical_levels": levels,
        "encoded_columns": encoded,
    }


def encode_features(
    frame: pd.DataFrame,
    schema: dict,
    output_path: Path | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    if [name for name in frame.columns if name != "TARGET"] != schema["feature_columns"]:
        raise ValueError("Predictor columns or their order differ from the trained schema.")
    shape = (len(frame), len(schema["encoded_columns"]))
    matrix = (
        np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=shape)
        if output_path is not None else np.empty(shape, dtype="float32")
    )
    numeric = schema["numeric_columns"]
    iterator = tqdm(numeric, desc="Encoding numeric columns", unit=" columns") if show_progress else numeric
    for column_index, name in enumerate(iterator):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype="float32", na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Predictor {name} contains infinity or exceeds float32 range.")
        matrix[:, column_index] = values
    next_index = len(numeric)
    unseen_counts: dict[str, int] = {}
    for name, levels in schema["categorical_levels"].items():
        values = frame[name].astype("string")
        unseen = values.notna() & ~values.isin(levels)
        unseen_counts[name] = int(unseen.sum())
        for level in levels:
            matrix[:, next_index] = values.eq(level).fillna(False).to_numpy(dtype="float32")
            next_index += 1
        matrix[:, next_index] = (values.isna() | unseen).to_numpy(dtype="float32")
        next_index += 1
    if next_index != matrix.shape[1]:
        raise RuntimeError("Encoded feature count does not match the schema.")
    if output_path is not None:
        matrix.flush()
    return matrix, unseen_counts


def validation_weights(months: pd.Series, shares: dict[int, float] | None) -> np.ndarray | None:
    if shares is None:
        return None
    counts = months.value_counts()
    if set(counts.index) != set(shares) or not np.isclose(sum(shares.values()), 1.0):
        raise ValueError("Validation month shares do not match the available months.")
    per_row = {month: share / int(counts[month]) for month, share in shares.items()}
    return months.map(per_row).to_numpy(dtype="float64")


def create_run_directory(base: Path, prefix: str = "run") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        stem = f"{prefix}_{stamp}"
        candidate = base / (stem if suffix == 0 else f"{stem}_{suffix}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique output directory.")
