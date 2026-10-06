"""Read chronological data and build train-only linear-model features."""

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
    202306, 202307, 202308, 202309, 202310, 202311, 202312,
    202401, 202402, 202403, 202404, 202405,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    train_months: tuple[int, ...]
    valid_months: tuple[int, ...]
    month_shares: dict[int, float] | None


def experiment_from_config(name: str, cfg: DictConfig) -> Experiment:
    spec = cfg.protocols[name]
    shares = spec.validation_month_shares
    result = Experiment(
        name,
        tuple(int(month) for month in spec.train_months),
        tuple(int(month) for month in spec.validation_months),
        None if shares is None else {int(month): float(share) for month, share in shares.items()},
    )
    if not result.train_months or not result.valid_months:
        raise ValueError("Training and validation months must be nonempty.")
    if len(set(result.train_months)) != len(result.train_months) or len(set(result.valid_months)) != len(result.valid_months):
        raise ValueError("Months must be unique within each split.")
    if set(result.train_months) & set(result.valid_months) or max(result.train_months) >= min(result.valid_months):
        raise ValueError("Training months must strictly precede validation months.")
    if result.month_shares is not None and (
        set(result.month_shares) != set(result.valid_months)
        or any(value <= 0 for value in result.month_shares.values())
        or not np.isclose(sum(result.month_shares.values()), 1.0)
    ):
        raise ValueError("Validation month shares must be positive, complete, and sum to one.")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path, chunk_rows: int, require_all_months: bool = True) -> pd.DataFrame:
    if not path.is_file() or chunk_rows < 1:
        raise ValueError("The source CSV must exist and data.chunk_rows must be positive.")
    chunks: list[pd.DataFrame] = []
    with tqdm(desc="Reading CSV", unit=" rows", position=0) as progress:
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            chunks.append(chunk)
            progress.update(len(chunk))
    if not chunks:
        raise ValueError("The source CSV is empty.")
    frame = pd.concat(chunks, ignore_index=True)
    if not frame.columns.is_unique or not {"LNMON", "TARGET"}.issubset(frame.columns):
        raise ValueError("The CSV needs unique columns including LNMON and TARGET.")
    if len(frame.columns) < 3:
        raise ValueError("No predictor columns were found.")
    if frame[["LNMON", "TARGET"]].isna().any(axis=None):
        raise ValueError("LNMON and TARGET cannot be missing.")
    frame["LNMON"] = pd.to_numeric(frame["LNMON"], errors="raise")
    frame["TARGET"] = pd.to_numeric(frame["TARGET"], errors="raise")
    if not frame["LNMON"].isin(ALL_MONTHS).all() or not frame["TARGET"].isin((0, 1)).all():
        raise ValueError("LNMON must be an expected month and TARGET must be binary.")
    if require_all_months and set(frame["LNMON"].unique()) != set(ALL_MONTHS):
        raise ValueError("The source CSV lacks one or more required months.")
    frame["LNMON"] = frame["LNMON"].astype("int32")
    frame["TARGET"] = frame["TARGET"].astype("int8")
    for name in tqdm(frame.select_dtypes(include="number").columns, desc="Compacting columns", unit=" columns"):
        if name in ("LNMON", "TARGET"):
            continue
        series = frame[name]
        if pd.api.types.is_integer_dtype(series):
            frame[name] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            observed = series.dropna().to_numpy(dtype="float64", copy=False)
            if not len(observed):
                frame[name] = series.astype("float32")
            elif (
                np.isfinite(observed).all()
                and observed.min() >= -(2**31)
                and observed.max() <= 2**31 - 1
                and np.equal(observed, np.trunc(observed)).all()
            ):
                frame[name] = series.astype("Int32")
    return frame


def _column_layout(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    columns = [name for name in frame.columns if name != "TARGET"]
    categorical = [
        name for name in columns
        if pd.api.types.is_object_dtype(frame[name])
        or pd.api.types.is_string_dtype(frame[name])
        or isinstance(frame[name].dtype, pd.CategoricalDtype)
    ]
    numeric = [name for name in columns if name not in categorical]
    return columns, numeric, categorical


def fit_schema(
    frame: pd.DataFrame, train_indices: np.ndarray,
    position: int = 0, label: str = "",
) -> dict:
    """Fit medians, scales, and text levels using training rows only."""

    columns, numeric, categorical = _column_layout(frame)
    numeric_stats = {}
    missing_columns = []
    for name in tqdm(
        numeric, desc=f"{label} fitting numerics".strip(),
        unit=" columns", position=position, leave=False,
    ):
        values = pd.to_numeric(frame[name].iloc[train_indices], errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        ).copy()
        if np.isinf(values).any():
            raise ValueError(f"Predictor {name} contains infinity.")
        missing = np.isnan(values)
        if missing.any():
            missing_columns.append(name)
        observed = values[~missing]
        median = float(np.median(observed)) if len(observed) else 0.0
        if missing.any():
            values[missing] = median
        mean = float(np.mean(values))
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        numeric_stats[name] = {"median": median, "mean": mean, "scale": scale}
    levels = {
        name: sorted(frame[name].iloc[train_indices].astype("string").dropna().unique().tolist())
        for name in categorical
    }
    encoded = list(numeric)
    encoded.extend(f"{name}=<MISSING>" for name in missing_columns)
    for name in categorical:
        encoded.extend(f"{name}={level}" for level in levels[name])
        encoded.append(f"{name}=<MISSING_OR_UNKNOWN>")
    return {
        "feature_columns": columns,
        "numeric_columns": numeric,
        "categorical_levels": levels,
        "numeric_stats": numeric_stats,
        "missing_indicator_columns": missing_columns,
        "encoded_columns": encoded,
    }


def transform_to_memmap(
    frame: pd.DataFrame,
    indices: np.ndarray,
    schema: dict,
    path: Path,
    show_progress: bool = False,
    position: int = 0,
    label: str = "",
) -> tuple[np.memmap, dict[str, int]]:
    if [name for name in frame.columns if name != "TARGET"] != schema["feature_columns"]:
        raise ValueError("Predictor columns or their order differ from the saved schema.")
    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype="float32", shape=(len(indices), len(schema["encoded_columns"]))
    )
    numeric = schema["numeric_columns"]
    missing_indices = {name: len(numeric) + i for i, name in enumerate(schema["missing_indicator_columns"])}
    iterator = (
        tqdm(
            numeric, desc=f"{label} encoding numerics".strip(),
            unit=" columns", position=position, leave=False,
        ) if show_progress else numeric
    )
    for column_index, name in enumerate(iterator):
        values = pd.to_numeric(frame[name].iloc[indices], errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        ).copy()
        if np.isinf(values).any():
            raise ValueError(f"Predictor {name} contains infinity.")
        missing = np.isnan(values)
        if name in missing_indices:
            matrix[:, missing_indices[name]] = missing.astype("float32")
        stat = schema["numeric_stats"][name]
        if missing.any():
            values[missing] = stat["median"]
        values -= stat["mean"]
        values /= stat["scale"]
        if not np.isfinite(values).all() or np.max(np.abs(values), initial=0) > np.finfo("float32").max:
            raise ValueError(f"Predictor {name} cannot be represented as finite float32.")
        matrix[:, column_index] = values.astype("float32")
    next_index = len(numeric) + len(missing_indices)
    unseen_counts = {}
    for name, levels in schema["categorical_levels"].items():
        values = frame[name].iloc[indices].astype("string")
        unseen = values.notna() & ~values.isin(levels)
        unseen_counts[name] = int(unseen.sum())
        for level in levels:
            matrix[:, next_index] = values.eq(level).fillna(False).to_numpy(dtype="float32")
            next_index += 1
        matrix[:, next_index] = (values.isna() | unseen).to_numpy(dtype="float32")
        next_index += 1
    if next_index != matrix.shape[1]:
        raise RuntimeError("Encoded feature count differs from the schema.")
    matrix.flush()
    return matrix, unseen_counts


def validation_weights(months: np.ndarray, shares: dict[int, float] | None) -> np.ndarray | None:
    if shares is None:
        return None
    observed, inverse, counts = np.unique(months, return_inverse=True, return_counts=True)
    if set(observed) != set(shares) or not np.isclose(sum(shares.values()), 1.0):
        raise ValueError("Validation months do not match their configured shares.")
    per_month = np.array(
        [shares[int(month)] / int(count) for month, count in zip(observed, counts)],
        dtype="float64",
    )
    return per_month[inverse]


def create_run_directory(base: Path, prefix: str = "run") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = f"{prefix}_{stamp}" + (f"_{suffix}" if suffix else "")
        candidate = base / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate a unique output directory.")
