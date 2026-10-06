"""Batched GaussianNB scoring, smoothing selection, and checkpoint I/O."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from tqdm.auto import tqdm

from .data import file_sha256, transform_batch


HISTORY_COLUMNS = ["update", "train_rows", "valid_error"]


def ordered_indices(months: np.ndarray, selected: tuple[int, ...]) -> np.ndarray:
    indices = np.flatnonzero(np.isin(months, selected))
    return indices[np.argsort(months[indices], kind="stable")]


def batch_slices(indices: np.ndarray, batch_rows: int):
    if batch_rows < 1:
        raise ValueError("training.batch_rows must be positive.")
    for start in range(0, len(indices), batch_rows):
        yield indices[start:start + batch_rows]


def predict_probabilities(
    model: GaussianNB, frame: pd.DataFrame, indices: np.ndarray,
    schema: dict, batch_rows: int, show_progress: bool = False,
) -> np.ndarray:
    if not len(indices):
        raise ValueError("Prediction rows are empty.")
    output = np.empty(len(indices), dtype="float64")
    iterator = batch_slices(np.arange(len(indices)), batch_rows)
    if show_progress:
        iterator = tqdm(iterator, total=(len(indices) + batch_rows - 1) // batch_rows,
                        desc="Predicting validation", unit=" batches")
    for positions in iterator:
        features = transform_batch(frame.iloc[indices[positions]], schema)
        output[positions] = model.predict_proba(features)[:, 1]
    if not np.isfinite(output).all():
        raise ValueError("GaussianNB returned nonfinite probabilities.")
    return output


def encode_to_memmap(
    frame: pd.DataFrame, indices: np.ndarray, schema: dict,
    batch_rows: int, path: Path, label: str,
) -> np.memmap:
    """Encode validation once, then reuse its disk-backed matrix at each update."""

    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype="float64",
        shape=(len(indices), len(schema["encoded_columns"])),
    )
    iterator = batch_slices(np.arange(len(indices)), batch_rows)
    iterator = tqdm(iterator, total=(len(indices) + batch_rows - 1) // batch_rows,
                    desc=f"Encoding {label} validation", unit=" batches")
    for positions in iterator:
        matrix[positions] = transform_batch(frame.iloc[indices[positions]], schema)
    matrix.flush()
    return matrix


def predict_matrix_probabilities(model: GaussianNB, matrix: np.ndarray,
                                 batch_rows: int) -> np.ndarray:
    output = np.empty(len(matrix), dtype="float64")
    for positions in batch_slices(np.arange(len(matrix)), batch_rows):
        output[positions] = model.predict_proba(matrix[positions])[:, 1]
    if not np.isfinite(output).all():
        raise ValueError("GaussianNB returned nonfinite probabilities.")
    return output


def misclassification(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None = None, threshold: float = 0.5,
) -> float:
    if not 0 < threshold < 1 or not len(target) or len(target) != len(probability):
        raise ValueError("Invalid target, probability, or classification threshold.")
    return float(np.average((probability > threshold) != target, weights=weights))


def choose_var_smoothing(
    frame: pd.DataFrame, months: np.ndarray, target: np.ndarray,
    train_months: tuple[int, ...], candidates: list[float], holdout_months: int,
    batch_rows: int, threshold: float, label: str,
) -> tuple[float, list[dict]]:
    """Select smoothing on the final training month(s), with an inner-only schema."""

    if holdout_months < 1 or holdout_months >= len(train_months):
        raise ValueError("The tuning holdout must leave at least one training month.")
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("var_smoothing candidates must be positive and finite.")
    inner_months = train_months[:-holdout_months]
    holdout = train_months[-holdout_months:]
    inner_indices = ordered_indices(months, inner_months)
    holdout_indices = ordered_indices(months, holdout)
    if not len(inner_indices) or not len(holdout_indices) or np.unique(target[inner_indices]).size != 2:
        raise ValueError("Inner training needs both classes and a nonempty holdout.")
    from .data import fit_schema

    schema = fit_schema(frame, inner_indices, f"{label} tuning")
    models = [GaussianNB(var_smoothing=float(value)) for value in candidates]
    batches = list(batch_slices(inner_indices, batch_rows))
    for batch_number, indices in enumerate(tqdm(batches, desc=f"Tuning {label}", unit=" batches")):
        features = transform_batch(frame.iloc[indices], schema)
        for model in models:
            model.partial_fit(features, target[indices], classes=np.array([0, 1]) if batch_number == 0 else None)
    errors = np.zeros(len(models), dtype="float64")
    holdout_target = target[holdout_indices]
    for positions in batch_slices(np.arange(len(holdout_indices)), batch_rows):
        features = transform_batch(frame.iloc[holdout_indices[positions]], schema)
        for index, model in enumerate(models):
            probabilities = model.predict_proba(features)[:, 1]
            if not np.isfinite(probabilities).all():
                raise ValueError("GaussianNB returned nonfinite tuning probabilities.")
            errors[index] += np.count_nonzero((probabilities > threshold) != holdout_target[positions])
    scores = errors / len(holdout_indices)
    results = [
        {"var_smoothing": float(value), "holdout_error": float(score)}
        for value, score in zip(candidates, scores)
    ]
    winner = min(results, key=lambda row: (row["holdout_error"], row["var_smoothing"]))
    return winner["var_smoothing"], results


def save_checkpoint(root: Path, model: GaussianNB, schema: dict, settings: dict,
                    history: list[dict]) -> Path:
    """Publish a complete checkpoint through an atomic directory rename."""

    root.mkdir(parents=True, exist_ok=True)
    update = len(history)
    if not update:
        raise ValueError("A checkpoint needs at least one update.")
    destination = root / f"update_{update:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=root))
    try:
        with (temporary / "model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "schema": schema}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(temporary / "history.csv", index=False)
        record = {
            "settings": settings,
            "checkpoint_update": update,
            "model_sha256": file_sha256(temporary / "model.pkl"),
            "history_sha256": file_sha256(temporary / "history.csv"),
        }
        (temporary / "state.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(path: Path) -> tuple[dict, GaussianNB, dict, list[dict]]:
    directory = path if path.is_dir() else path.parent
    record = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if (file_sha256(directory / "model.pkl") != record["model_sha256"]
            or file_sha256(directory / "history.csv") != record["history_sha256"]):
        raise ValueError("Checkpoint checksums do not match.")
    with (directory / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    history_frame = pd.read_csv(directory / "history.csv")
    if (list(history_frame.columns) != HISTORY_COLUMNS
            or history_frame["update"].tolist() != list(range(1, record["checkpoint_update"] + 1))
            or len(history_frame) != record["checkpoint_update"]):
        raise ValueError("Checkpoint history is incomplete.")
    if not isinstance(artifact.get("model"), GaussianNB) or not artifact.get("schema"):
        raise ValueError("Checkpoint model is invalid.")
    return record["settings"], artifact["model"], artifact["schema"], history_frame.to_dict("records")


def save_final_model(path: Path, model: GaussianNB, schema: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"model": model, "schema": schema}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_final_model(path: Path) -> tuple[GaussianNB, dict]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    model = artifact.get("model")
    schema = artifact.get("schema")
    if not isinstance(model, GaussianNB) or not isinstance(schema, dict):
        raise ValueError("The final model artifact is incomplete.")
    return model, schema
