"""Public-API, resumable binary AdaBoost-SAMME training primitives."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from . import COMPATIBLE_WORKFLOW_VERSIONS


def tree_predict(tree: DecisionTreeClassifier, features: np.ndarray) -> np.ndarray:
    """Predict with the public tree API while reusing a validated float32 matrix."""

    leaves = tree.apply(features, check_input=False)
    return tree.classes_.take(np.argmax(tree.tree_.value[leaves, 0, :], axis=1))


@dataclass
class SAMMEModel:
    max_depth: int
    n_estimators: int
    learning_rate: float
    random_state: int
    estimators: list[DecisionTreeClassifier] = field(default_factory=list)
    estimator_weights: list[float] = field(default_factory=list)
    estimator_errors: list[float] = field(default_factory=list)

    @property
    def fitted_iterations(self) -> int:
        return len(self.estimators)

    def predict_margin(self, features: np.ndarray, workers: int = 1) -> np.ndarray:
        if not self.estimators:
            raise ValueError("The model has no fitted trees.")
        matrix = np.ascontiguousarray(features, dtype=np.float32)
        margin = np.zeros(len(matrix), dtype=np.float64)
        if workers == 1:
            predictions = (tree_predict(tree, matrix) for tree in self.estimators)
            for alpha, predicted in zip(self.estimator_weights, predictions):
                margin += alpha * (2 * predicted - 1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for alpha, predicted in zip(
                    self.estimator_weights,
                    pool.map(lambda tree: tree_predict(tree, matrix), self.estimators),
                ):
                    margin += alpha * (2 * predicted - 1)
        return margin

    def predict_proba(self, features: np.ndarray, workers: int = 1) -> np.ndarray:
        margin = self.predict_margin(features, workers=workers)
        decision = 2.0 * margin / np.sum(self.estimator_weights)
        positive = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: np.ndarray, workers: int = 1) -> np.ndarray:
        return (self.predict_margin(features, workers=workers) > 0).astype(np.int8)


@dataclass
class TrainingSnapshot:
    model: SAMMEModel
    sample_weight: np.ndarray
    rng_state: tuple
    valid_margin: np.ndarray
    history: list[float]
    stopping_reason: str | None = None


def initial_snapshot(model: SAMMEModel, train_rows: int, valid_rows: int) -> TrainingSnapshot:
    rng = np.random.RandomState(model.random_state)
    return TrainingSnapshot(
        model=model,
        sample_weight=np.full(train_rows, 1.0 / train_rows, dtype=np.float64),
        rng_state=rng.get_state(),
        valid_margin=np.zeros(valid_rows, dtype=np.float64),
        history=[],
    )


def boost_one(
    snapshot: TrainingSnapshot,
    train_features: np.ndarray,
    train_target: np.ndarray,
    valid_features: np.ndarray,
    valid_target: np.ndarray,
    valid_weights: np.ndarray | None,
) -> float | None:
    """Fit one SAMME tree, update training weights, and score ensemble 0/1 error."""

    model = snapshot.model
    if snapshot.stopping_reason is not None or model.fitted_iterations >= model.n_estimators:
        raise ValueError("This snapshot cannot accept another boosting iteration.")
    rng = np.random.RandomState()
    rng.set_state(snapshot.rng_state)
    tree = DecisionTreeClassifier(
        max_depth=model.max_depth,
        random_state=int(rng.randint(np.iinfo(np.int32).max)),
    )
    sample_weight = np.maximum(snapshot.sample_weight, np.finfo(np.float64).eps)
    tree.fit(train_features, train_target, sample_weight=sample_weight)
    incorrect = tree_predict(tree, train_features) != train_target
    error = float(np.average(incorrect, weights=sample_weight))
    if error >= 0.5:
        if not model.estimators:
            raise ValueError("The first weak learner is no better than random guessing.")
        snapshot.stopping_reason = "weak_learner_error_at_least_half"
        return None
    if error <= 0.0:
        alpha = 1.0
        snapshot.stopping_reason = "perfect_weak_learner"
    else:
        alpha = float(model.learning_rate * np.log((1.0 - error) / error))
    model.estimators.append(tree)
    model.estimator_weights.append(alpha)
    model.estimator_errors.append(error)
    snapshot.valid_margin += alpha * (2 * tree_predict(tree, valid_features) - 1)
    wrong = (snapshot.valid_margin > 0) != valid_target
    valid_error = float(np.average(wrong, weights=valid_weights))
    snapshot.history.append(valid_error)
    if snapshot.stopping_reason is None and model.fitted_iterations < model.n_estimators:
        sample_weight = np.exp(np.log(sample_weight) + alpha * incorrect)
        sample_weight_sum = sample_weight.sum()
        if not np.isfinite(sample_weight_sum) or sample_weight_sum <= 0:
            snapshot.stopping_reason = "invalid_training_weights"
        else:
            snapshot.sample_weight = sample_weight / sample_weight_sum
    snapshot.rng_state = rng.get_state()
    return valid_error


def save_checkpoint(root: Path, snapshot: TrainingSnapshot, settings: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    iteration = snapshot.model.fitted_iterations
    if iteration != len(snapshot.history):
        raise RuntimeError("Checkpoint history and model lengths differ.")
    destination = root / f"round_{iteration:06d}"
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=root))
    try:
        with (temporary / "snapshot.pkl").open("wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame({
            "iteration": np.arange(1, iteration + 1),
            "valid_error": snapshot.history,
        }).to_csv(temporary / "history.csv", index=False)
        (temporary / "state.json").write_text(
            json.dumps({
                **settings,
                "checkpoint_iteration": iteration,
                "stopping_reason": snapshot.stopping_reason,
            }, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(path: Path) -> tuple[dict, TrainingSnapshot]:
    directory = path if path.is_dir() else path.parent
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if state.get("workflow_version") not in COMPATIBLE_WORKFLOW_VERSIONS:
        raise ValueError("The checkpoint workflow version differs from this implementation.")
    with (directory / "snapshot.pkl").open("rb") as handle:
        snapshot = pickle.load(handle)
    history = pd.read_csv(directory / "history.csv")
    iteration = int(state["checkpoint_iteration"])
    if (
        not isinstance(snapshot, TrainingSnapshot)
        or snapshot.model.fitted_iterations != iteration
        or len(snapshot.history) != iteration
        or list(history.columns) != ["iteration", "valid_error"]
        or history["iteration"].tolist() != list(range(1, iteration + 1))
        or not np.allclose(history["valid_error"], snapshot.history, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("The checkpoint model or history is incomplete.")
    return state, snapshot


def save_final_model(path: Path, model: SAMMEModel) -> None:
    with tempfile.NamedTemporaryFile(mode="wb", prefix=".model_", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_final_model(path: Path) -> SAMMEModel:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, SAMMEModel) or not model.estimators:
        raise ValueError("The file does not contain a fitted AdaBoost SAMME model.")
    return model
