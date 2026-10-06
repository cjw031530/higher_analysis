"""Train, score, save, and resume the three native tree models."""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


MEMBERS = ("catboost", "lightgbm", "xgboost")


@dataclass
class FeatureSet:
    native: pd.DataFrame
    encoded: pd.DataFrame
    target: np.ndarray | None
    categorical: list[str]
    catboost_pool: catboost.Pool | None = None
    lightgbm_dataset: lgb.Dataset | None = None


@dataclass
class MemberModels:
    catboost: catboost.CatBoostClassifier
    lightgbm: lgb.Booster
    xgboost: xgb.XGBClassifier


def classification_error(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None = None, threshold: float = 0.5,
) -> float:
    """Weighted zero-one loss; ties at the threshold predict class zero."""
    if len(target) != len(probability) or not 0 < threshold < 1:
        raise ValueError("Targets, probabilities, or threshold are invalid.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    return float(np.average((probability > threshold) != target, weights=weights))


def average_probabilities(scores: dict[str, np.ndarray]) -> np.ndarray:
    if set(scores) != set(MEMBERS):
        raise ValueError("The ensemble requires exactly three member probabilities.")
    return np.mean(np.stack([scores[name] for name in MEMBERS], axis=0), axis=0)


def _fit_one(
    name: str, train: FeatureSet, rounds: int, params: dict,
    threads: int, previous: MemberModels | None,
) -> object:
    if name == "catboost":
        model = catboost.CatBoostClassifier(
            iterations=rounds, loss_function="Logloss", thread_count=threads,
            allow_writing_files=False, use_best_model=False, **params,
        )
        if train.catboost_pool is None:
            train.catboost_pool = catboost.Pool(
                train.native, label=train.target, cat_features=train.categorical,
            )
        model.fit(train.catboost_pool, init_model=None if previous is None else previous.catboost, verbose=False)
        return model
    if name == "lightgbm":
        parameters = {
            **params, "objective": "binary", "metric": "binary_error",
            "num_threads": threads, "feature_pre_filter": False,
        }
        if train.lightgbm_dataset is None:
            train.lightgbm_dataset = lgb.Dataset(
                train.encoded, label=train.target,
                categorical_feature=train.categorical, free_raw_data=False,
            )
        return lgb.train(
            parameters, train.lightgbm_dataset, num_boost_round=rounds,
            init_model=None if previous is None else previous.lightgbm,
            keep_training_booster=True,
        )
    if name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=rounds, objective="binary:logistic", eval_metric="error",
            enable_categorical=True, n_jobs=threads, **params,
        )
        model.fit(
            train.encoded, train.target,
            xgb_model=None if previous is None else previous.xgboost.get_booster(),
            verbose=False,
        )
        return model
    raise ValueError(f"Unknown member: {name}")


def fit_members(
    train: FeatureSet, rounds: int, member_params: dict[str, dict],
    threads: int, previous: MemberModels | None = None,
    parallel: bool = True,
) -> MemberModels:
    if rounds < 1 or threads < 1:
        raise ValueError("Round and thread counts must be positive.")
    if parallel:
        with ThreadPoolExecutor(max_workers=len(MEMBERS)) as executor:
            futures = {
                name: executor.submit(_fit_one, name, train, rounds, member_params[name], threads, previous)
                for name in MEMBERS
            }
            trained = {name: future.result() for name, future in futures.items()}
    else:
        trained = {
            name: _fit_one(name, train, rounds, member_params[name], threads, previous)
            for name in MEMBERS
        }
    return MemberModels(**trained)


def predict_members(
    models: MemberModels, features: FeatureSet, threads: int,
    rounds: int | None = None, parallel: bool = True,
) -> dict[str, np.ndarray]:
    def predict(name: str) -> np.ndarray:
        if name == "catboost":
            if features.catboost_pool is None:
                features.catboost_pool = catboost.Pool(
                    features.native, cat_features=features.categorical,
                )
            result = models.catboost.predict_proba(
                features.catboost_pool, ntree_end=0 if rounds is None else rounds,
                thread_count=threads,
            )[:, 1]
        elif name == "lightgbm":
            result = models.lightgbm.predict(
                features.encoded, num_iteration=rounds, num_threads=threads,
            )
        else:
            result = models.xgboost.predict_proba(
                features.encoded,
                iteration_range=(0, rounds) if rounds is not None else None,
            )[:, 1]
        return np.asarray(result, dtype="float64")

    if parallel:
        with ThreadPoolExecutor(max_workers=len(MEMBERS)) as executor:
            futures = {name: executor.submit(predict, name) for name in MEMBERS}
            return {name: future.result() for name, future in futures.items()}
    return {name: predict(name) for name in MEMBERS}


def boosted_rounds(models: MemberModels) -> dict[str, int]:
    return {
        "catboost": int(models.catboost.tree_count_),
        "lightgbm": int(models.lightgbm.current_iteration()),
        "xgboost": int(models.xgboost.get_booster().num_boosted_rounds()),
    }


def save_models(path: Path, models: MemberModels) -> None:
    path.mkdir(parents=True, exist_ok=False)
    models.catboost.save_model(str(path / "catboost.cbm"))
    models.lightgbm.save_model(str(path / "lightgbm.txt"))
    models.xgboost.save_model(path / "xgboost.json")


def load_models(path: Path, threads: int = 1) -> MemberModels:
    model_path = path if path.is_dir() else path.parent
    cat = catboost.CatBoostClassifier()
    cat.load_model(str(model_path / "catboost.cbm"))
    light = lgb.Booster(model_file=str(model_path / "lightgbm.txt"))
    xg = xgb.XGBClassifier(enable_categorical=True, n_jobs=threads)
    xg.load_model(model_path / "xgboost.json")
    return MemberModels(cat, light, xg)


def save_checkpoint(
    root: Path, models: MemberModels, settings: dict,
    history: list[dict], completed_rounds: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"round_{completed_rounds:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=root))
    try:
        save_models(temporary / "model", models)
        pd.DataFrame(history).to_csv(temporary / "history.csv", index=False)
        (temporary / "state.json").write_text(
            json.dumps({**settings, "completed_rounds": completed_rounds}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(path: Path, threads: int) -> tuple[dict, MemberModels, list[dict]]:
    directory = path if path.is_dir() else path.parent
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    history = pd.read_csv(directory / "history.csv").to_dict("records")
    models = load_models(directory / "model", threads)
    rounds = int(state["completed_rounds"])
    if not history or int(history[-1]["round"]) != rounds:
        raise ValueError("Checkpoint history does not end at its saved round.")
    if any(value != rounds for value in boosted_rounds(models).values()):
        raise ValueError("Checkpoint members have different round counts.")
    return state, models, history
