"""Persist every SAGA optimizer array at completed epoch boundaries.

The compiled scikit-learn SAGA kernel is called directly. Each epoch starts a
new, deterministically seeded dataset; the complete gradient table and running
gradients survive every call and are serialized for exact workflow replay.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import sag32
from sklearn.utils._seq_dataset import ArrayDataset32
from sklearn.utils.extmath import row_norms

from .data import file_sha256


@dataclass
class SAGAState:
    coef: np.ndarray
    intercept: np.ndarray
    sum_gradient: np.ndarray
    intercept_sum_gradient: np.ndarray
    gradient_memory: np.ndarray
    seen: np.ndarray
    num_seen: int
    epoch: int
    max_squared_sum: float
    step_size: float
    random_state: int


def regularization(C: float, l1_ratio: float, n_samples: int) -> tuple[float, float]:
    """Return scikit-learn's per-sample L2 and L1 coefficients."""

    if C <= 0 or not 0 <= l1_ratio <= 1 or n_samples < 1:
        raise ValueError("C must be positive, l1_ratio in [0, 1], and training nonempty.")
    alpha = (1.0 / C) * (1.0 - l1_ratio) / n_samples
    beta = (1.0 / C) * l1_ratio / n_samples
    return alpha, beta


def initial_state(X: np.ndarray, C: float, l1_ratio: float, random_state: int) -> SAGAState:
    if X.dtype != np.float32 or not X.flags.c_contiguous or X.ndim != 2:
        raise ValueError("SAGA requires a C-contiguous float32 matrix.")
    if random_state < 0:
        raise ValueError("model.random_state must be nonnegative.")
    n_samples, n_features = X.shape
    if n_samples < 1 or n_features < 1:
        raise ValueError("The training matrix must be nonempty.")
    max_squared_sum = float(row_norms(X, squared=True).max())
    alpha, _ = regularization(C, l1_ratio, n_samples)
    step_size = float(get_auto_step_size(
        max_squared_sum, alpha, "log", True, n_samples=n_samples, is_saga=True
    ))
    return SAGAState(
        coef=np.zeros((n_features, 1), dtype="float32"),
        intercept=np.zeros(1, dtype="float32"),
        sum_gradient=np.zeros((n_features, 1), dtype="float32"),
        intercept_sum_gradient=np.zeros(1, dtype="float32"),
        gradient_memory=np.zeros((n_samples, 1), dtype="float32"),
        seen=np.zeros(n_samples, dtype="int32"),
        num_seen=0,
        epoch=0,
        max_squared_sum=max_squared_sum,
        step_size=step_size,
        random_state=int(random_state),
    )


def check_state(state: SAGAState, X: np.ndarray, random_state: int) -> None:
    n_samples, n_features = X.shape
    if (
        state.coef.shape != (n_features, 1)
        or state.sum_gradient.shape != (n_features, 1)
        or state.gradient_memory.shape != (n_samples, 1)
        or state.seen.shape != (n_samples,)
        or state.intercept.shape != (1,)
        or state.intercept_sum_gradient.shape != (1,)
        or state.random_state != random_state
        or state.max_squared_sum != float(row_norms(X, squared=True).max())
    ):
        raise ValueError("Checkpoint optimizer state is incompatible with the training matrix.")
    if any(array.dtype != np.float32 for array in (
        state.coef, state.intercept, state.sum_gradient,
        state.intercept_sum_gradient, state.gradient_memory,
    )) or state.seen.dtype != np.int32:
        raise ValueError("Checkpoint optimizer arrays have incompatible dtypes.")


def _epoch_seed(random_state: int, epoch: int) -> int:
    seed = int(np.random.SeedSequence([random_state, epoch + 1]).generate_state(1)[0])
    return seed or 1


def run_epoch(
    state: SAGAState,
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    l1_ratio: float,
) -> float:
    """Advance one complete epoch and return relative coefficient change."""

    if X.dtype != np.float32 or y.dtype != np.float32 or not X.flags.c_contiguous:
        raise ValueError("SAGA input arrays must be contiguous float32.")
    if y.shape != (X.shape[0],) or not np.isin(y, (0, 1)).all():
        raise ValueError("SAGA target must contain binary labels for every training row.")
    previous = state.coef.copy()
    weights = np.ones(len(y), dtype="float32")
    dataset = ArrayDataset32(X, y, weights, seed=_epoch_seed(state.random_state, state.epoch))
    alpha, beta = regularization(C, l1_ratio, len(y))
    state.num_seen, completed = sag32(
        dataset, state.coef, state.intercept, len(y), X.shape[1], 1,
        0.0, 1, "log", state.step_size, alpha, beta,
        state.sum_gradient, state.gradient_memory, state.seen,
        state.num_seen, True, state.intercept_sum_gradient, 1.0, True, False,
    )
    if completed != 1:
        raise RuntimeError("The compiled SAGA kernel did not finish exactly one epoch.")
    state.epoch += 1
    maximum = float(np.max(np.abs(state.coef)))
    change = float(np.max(np.abs(state.coef - previous)))
    return change / maximum if maximum else change


def predict_probability(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    if X.shape[1] != coef.shape[0]:
        raise ValueError("Feature count does not match the saved coefficients.")
    scores = X @ coef[:, 0] + intercept[0]
    return expit(scores.astype("float64"))


def _state_arrays(state: SAGAState) -> dict:
    return {
        "coef": state.coef,
        "intercept": state.intercept,
        "sum_gradient": state.sum_gradient,
        "intercept_sum_gradient": state.intercept_sum_gradient,
        "gradient_memory": state.gradient_memory,
        "seen": state.seen,
        "num_seen": np.array(state.num_seen, dtype="int64"),
        "epoch": np.array(state.epoch, dtype="int64"),
        "max_squared_sum": np.array(state.max_squared_sum, dtype="float64"),
        "step_size": np.array(state.step_size, dtype="float64"),
        "random_state": np.array(state.random_state, dtype="int64"),
    }


def save_checkpoint(root: Path, state: SAGAState, settings: dict, history: list[dict]) -> Path:
    """Publish a checksum-verified optimizer snapshot atomically."""

    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"epoch_{state.epoch:06d}"
    if destination.exists():
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=root))
    try:
        np.savez_compressed(temporary / "optimizer_state.npz", **_state_arrays(state))
        pd.DataFrame(history, columns=["epoch", "valid_error", "relative_coef_change"]).to_csv(
            temporary / "history.csv", index=False
        )
        state_record = {
            "settings": settings,
            "checkpoint_epoch": state.epoch,
            "optimizer_sha256": file_sha256(temporary / "optimizer_state.npz"),
            "history_sha256": file_sha256(temporary / "history.csv"),
        }
        (temporary / "state.json").write_text(
            json.dumps(state_record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(path: Path) -> tuple[dict, SAGAState, list[dict]]:
    directory = path if path.is_dir() else path.parent
    record = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    optimizer_path = directory / "optimizer_state.npz"
    history_path = directory / "history.csv"
    if file_sha256(optimizer_path) != record["optimizer_sha256"] or file_sha256(history_path) != record["history_sha256"]:
        raise ValueError("Checkpoint checksum verification failed.")
    with np.load(optimizer_path, allow_pickle=False) as arrays:
        state = SAGAState(
            coef=arrays["coef"].copy(),
            intercept=arrays["intercept"].copy(),
            sum_gradient=arrays["sum_gradient"].copy(),
            intercept_sum_gradient=arrays["intercept_sum_gradient"].copy(),
            gradient_memory=arrays["gradient_memory"].copy(),
            seen=arrays["seen"].copy(),
            num_seen=int(arrays["num_seen"]),
            epoch=int(arrays["epoch"]),
            max_squared_sum=float(arrays["max_squared_sum"]),
            step_size=float(arrays["step_size"]),
            random_state=int(arrays["random_state"]),
        )
    history_frame = pd.read_csv(history_path)
    if list(history_frame.columns) != ["epoch", "valid_error", "relative_coef_change"]:
        raise ValueError("Checkpoint history columns are invalid.")
    history = history_frame.to_dict("records")
    if state.epoch != record["checkpoint_epoch"] or len(history) != state.epoch:
        raise ValueError("Checkpoint optimizer epoch and history disagree.")
    if history_frame["epoch"].tolist() != list(range(1, state.epoch + 1)):
        raise ValueError("Checkpoint history has missing epochs.")
    return record["settings"], state, history


def save_final_model(path: Path, schema: dict, state: SAGAState) -> None:
    """Store prediction inputs and coefficients, excluding optimizer memory."""

    artifact = {
        "schema": schema,
        "coef": state.coef[:, 0].copy(),
        "intercept": float(state.intercept[0]),
        "epochs": state.epoch,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_final_model(path: Path) -> dict:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not {"schema", "coef", "intercept", "epochs"}.issubset(model):
        raise ValueError("The saved Elastic Net model is incomplete.")
    return model
