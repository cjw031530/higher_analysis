"""Read the source CSV and build training-only feature schemas."""

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
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class Protocol:
    name: str
    train_months: tuple[int, ...]
    validation_months: tuple[int, ...]
    validation_month_shares: dict[int, float] | None


def protocol_from_config(name: str, cfg: DictConfig) -> Protocol:
    item = cfg.protocols[name]
    shares = item.validation_month_shares
    result = Protocol(
        name,
        tuple(int(month) for month in item.train_months),
        tuple(int(month) for month in item.validation_months),
        None if shares is None else {int(key): float(value) for key, value in shares.items()},
    )
    if not result.train_months or not result.validation_months:
        raise ValueError("Training and validation months must be nonempty.")
    if (len(set(result.train_months)) != len(result.train_months)
            or len(set(result.validation_months)) != len(result.validation_months)):
        raise ValueError("Months must be unique within a split.")
    if set(result.train_months) & set(result.validation_months):
        raise ValueError("Training and validation months must not overlap.")
    if max(result.train_months) >= min(result.validation_months):
        raise ValueError("All training months must precede validation months.")
    if result.validation_month_shares is not None and (
        set(result.validation_month_shares) != set(result.validation_months)
        or any(value <= 0 for value in result.validation_month_shares.values())
        or not np.isclose(sum(result.validation_month_shares.values()), 1.0)
    ):
        raise ValueError("Validation shares must be positive, complete, and sum to one.")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(
    path: Path, chunk_rows: int, require_all_months: bool = True,
    months_filter: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if chunk_rows < 1:
        raise ValueError("data.chunk_rows must be positive.")
    chunks: list[pd.DataFrame] = []
    observed: set[int] = set()
    with tqdm(desc="Reading CSV", unit=" rows") as progress:
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            if "LNMON" not in chunk:
                raise ValueError("The CSV needs LNMON.")
            original_rows = len(chunk)
            observed.update(pd.to_numeric(chunk["LNMON"], errors="raise").astype("int32").unique())
            if months_filter is not None:
                chunk = chunk.loc[chunk["LNMON"].isin(months_filter)]
            if not chunk.empty:
                chunks.append(chunk)
            progress.update(original_rows)
    if not chunks:
        raise ValueError("The source CSV is empty.")
    frame = pd.concat(chunks, ignore_index=True)
    if not frame.columns.is_unique or not {"LNMON", "TARGET"}.issubset(frame.columns):
        raise ValueError("The CSV needs unique columns including LNMON and TARGET.")
    if frame[["LNMON", "TARGET"]].isna().any(axis=None):
        raise ValueError("LNMON and TARGET must not be missing.")
    frame["LNMON"] = pd.to_numeric(frame["LNMON"], errors="raise").astype("int32")
    frame["TARGET"] = pd.to_numeric(frame["TARGET"], errors="raise").astype("int8")
    if not frame["TARGET"].isin((0, 1)).all():
        raise ValueError("TARGET must be binary 0 or 1.")
    if not observed.issubset(ALL_MONTHS):
        raise ValueError("The CSV contains a month outside the supported period.")
    if require_all_months and observed != set(ALL_MONTHS):
        raise ValueError(f"The CSV is missing months: {sorted(set(ALL_MONTHS) - observed)}")
    if len(frame.columns) < 3:
        raise ValueError("The CSV has no predictor columns.")
    for name in frame.select_dtypes(include="integer").columns:
        frame[name] = pd.to_numeric(frame[name], downcast="integer")
    return frame


def fit_schema(frame: pd.DataFrame) -> dict:
    """Derive category levels from the supplied training rows only."""
    columns = [name for name in frame.columns if name != "TARGET"]
    categorical = [
        name for name in columns
        if (pd.api.types.is_object_dtype(frame[name].dtype)
            or isinstance(frame[name].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(frame[name].dtype))
    ]
    levels = {
        name: sorted(frame[name].dropna().astype(str).unique().tolist())
        for name in categorical
    }
    return {
        "feature_columns": columns,
        "categorical_features": categorical,
        "categorical_levels": levels,
        "missing_category": MISSING_CATEGORY,
    }


def transform_features(frame: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return native categorical frames for CatBoost and LightGBM/XGBoost."""
    columns = schema["feature_columns"]
    if not set(columns).issubset(frame.columns):
        raise ValueError("Input data does not contain every trained predictor.")
    native = frame.loc[:, columns].copy()
    encoded = frame.loc[:, columns].copy()
    for name in columns:
        if name in schema["categorical_levels"]:
            values = frame[name].astype("string")
            if values.eq(MISSING_CATEGORY).any():
                raise ValueError(f"The reserved missing category occurs in {name}.")
            native[name] = values.fillna(MISSING_CATEGORY).astype(str)
            dtype = pd.CategoricalDtype(categories=schema["categorical_levels"][name])
            encoded[name] = values.mask(~values.isin(dtype.categories)).astype(dtype)
        else:
            native[name] = pd.to_numeric(native[name], errors="raise").astype("float64")
            encoded[name] = pd.to_numeric(encoded[name], errors="raise").astype("float64")
    return native, encoded


def validation_weights(months: np.ndarray, shares: dict[int, float] | None) -> np.ndarray | None:
    if shares is None:
        return None
    unique, inverse, counts = np.unique(months, return_inverse=True, return_counts=True)
    if set(unique) != set(shares) or not np.isclose(sum(shares.values()), 1.0):
        raise ValueError("Validation months do not match their configured shares.")
    per_month = np.array([shares[int(month)] for month in unique], dtype="float64") / counts
    return per_month[inverse]


def create_run_directory(base: Path, prefix: str = "run") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        stem = f"{prefix}_{stamp}" + (f"_{suffix}" if suffix else "")
        candidate = base / stem
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            pass
    raise RuntimeError("Unable to create a unique run directory.")
