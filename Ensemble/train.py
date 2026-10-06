"""Fit equal-weight ensembles with train-only round selection and checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import catboost
import hydra
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from Ensemble import WORKFLOW_VERSION
from Ensemble.data import (
    create_run_directory, file_sha256, fit_schema, load_data,
    protocol_from_config, transform_features, validation_weights,
)
from Ensemble.model import (
    FeatureSet, average_probabilities, boosted_rounds, classification_error,
    fit_members, load_checkpoint, predict_members, save_checkpoint, save_models,
)


def feature_set(frame: pd.DataFrame, schema: dict) -> FeatureSet:
    native, encoded = transform_features(frame, schema)
    target = frame["TARGET"].to_numpy(dtype="int8") if "TARGET" in frame else None
    return FeatureSet(native, encoded, target, schema["categorical_features"])


def round_grid(max_rounds: int, every: int) -> list[int]:
    return sorted(set(range(every, max_rounds + 1, every)) | {max_rounds})


def select_rounds(
    frame: pd.DataFrame, train_months: tuple[int, ...], holdout_months: int,
    max_rounds: int, selection_every: int, params: dict, threads: int,
    parallel: bool, threshold: float, label: str,
) -> tuple[int, list[dict]]:
    """Choose one common tree budget using only the last training month(s)."""
    if holdout_months < 1 or holdout_months >= len(train_months):
        raise ValueError("The inner holdout must be a strict subset of training months.")
    fit_months = train_months[:-holdout_months]
    holdout = train_months[-holdout_months:]
    fit_frame = frame.loc[frame["LNMON"].isin(fit_months)]
    holdout_frame = frame.loc[frame["LNMON"].isin(holdout)]
    if fit_frame.empty or holdout_frame.empty:
        raise ValueError("The inner training or holdout period has no rows.")
    schema = fit_schema(fit_frame)
    fit_features = feature_set(fit_frame, schema)
    holdout_features = feature_set(holdout_frame, schema)
    print(f"{label}: fitting inner models; holdout months={list(holdout)}")
    models = None
    rows: list[dict] = []
    completed = 0
    with tqdm(total=max_rounds, desc=f"{label} inner selection", unit=" trees") as progress:
        for round_number in round_grid(max_rounds, selection_every):
            models = fit_members(
                fit_features, round_number - completed, params, threads,
                previous=models, parallel=parallel,
            )
            completed = round_number
            probabilities = predict_members(models, holdout_features, threads, parallel=parallel)
            ensemble = average_probabilities(probabilities)
            record = {
                "round": round_number,
                "ensemble_error": classification_error(holdout_features.target, ensemble, threshold=threshold),
                **{
                    f"{name}_error": classification_error(holdout_features.target, score, threshold=threshold)
                    for name, score in probabilities.items()
                },
            }
            rows.append(record)
            progress.update(round_number - progress.n)
            progress.set_postfix_str(f"inner ensemble error={record['ensemble_error']:.5f}")
    selected = min(rows, key=lambda row: (row["ensemble_error"], row["round"]))
    print(f"{label}: selected {selected['round']} rounds from inner ensemble 0/1 loss={selected['ensemble_error']:.6f}")
    return int(selected["round"]), rows


class LivePlot:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        if enabled:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.figure, self.axis = plt.subplots(figsize=(9, 5))
            self.axis.set(
                title="Outer validation error (display only)",
                xlabel="Boosting rounds", ylabel="Misclassification rate",
            )
            self.axis.grid(alpha=0.25)
            plt.show(block=False)

    def update(self, history: list[dict], protocol: str) -> None:
        if not self.enabled:
            return
        self.axis.clear()
        for column, label in (
            ("valid_error", "equal ensemble"),
            ("catboost_error", "CatBoost"),
            ("lightgbm_error", "LightGBM"),
            ("xgboost_error", "XGBoost"),
        ):
            self.axis.plot([row["round"] for row in history], [row[column] for row in history], label=label)
        self.axis.set(title=f"{protocol} outer validation (display only)",
                      xlabel="Boosting rounds", ylabel="Misclassification rate")
        self.axis.grid(alpha=0.25)
        self.axis.legend()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)


def train_protocol(
    cfg: DictConfig, name: str, frame: pd.DataFrame, source: Path,
    source_hash: str, run_dir: Path, threads: int,
    resume: tuple[dict, object, list[dict]] | None = None,
    plot: LivePlot | None = None,
) -> dict:
    protocol = protocol_from_config(name, cfg)
    train_frame = frame.loc[frame["LNMON"].isin(protocol.train_months)]
    valid_frame = frame.loc[frame["LNMON"].isin(protocol.validation_months)]
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"{name} has empty training or validation rows.")
    if train_frame["TARGET"].nunique() != 2:
        raise ValueError(f"{name} training requires both target classes.")
    schema = fit_schema(train_frame)
    params = {
        member: OmegaConf.to_container(cfg.model[member], resolve=True)
        for member in ("catboost", "lightgbm", "xgboost")
    }
    threshold = float(cfg.model.decision_threshold)
    config_record = {
        "max_rounds": int(cfg.model.max_rounds),
        "selection_every": int(cfg.model.selection_every),
        "checkpoint_every": int(cfg.checkpoint.every),
        "inner_holdout_months": int(cfg.training.inner_holdout_months),
        "decision_threshold": threshold,
        "members": params,
        "threads_per_member": threads,
        "parallel_members": bool(cfg.runtime.parallel_members),
    }
    settings = {
        "workflow_version": WORKFLOW_VERSION,
        "experiment": name,
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "train_months": list(protocol.train_months),
        "validation_months": list(protocol.validation_months),
        "validation_month_shares": protocol.validation_month_shares,
        "train_rows": len(train_frame),
        "validation_rows": len(valid_frame),
        "schema": schema,
        "hyperparameters": config_record,
        "history_metric": "misclassification_rate",
        "history_is_weighted": protocol.validation_month_shares is not None,
        "versions": {
            "catboost": catboost.__version__, "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__, "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    settings = json.loads(json.dumps(settings, allow_nan=False))
    previous_models = None
    history: list[dict] = []
    starting_round = 0
    if resume is None:
        selected, inner_history = select_rounds(
            frame, protocol.train_months, int(cfg.training.inner_holdout_months),
            int(cfg.model.max_rounds), int(cfg.model.selection_every), params,
            threads, bool(cfg.runtime.parallel_members), threshold, name,
        )
    else:
        prior, previous_models, history = resume
        for key in settings:
            if prior.get(key) != settings[key]:
                raise ValueError(f"Checkpoint setting differs from this run: {key}")
        selected = int(prior["selected_rounds"])
        inner_history = prior["inner_selection_history"]
        starting_round = int(prior["completed_rounds"])
        if starting_round > selected:
            raise ValueError("Checkpoint exceeds the selected round count.")
        if plot is not None and history:
            plot.update(history, name)
    settings["selected_rounds"] = selected
    settings["inner_selection_history"] = inner_history
    train_features = feature_set(train_frame, schema)
    valid_features = feature_set(valid_frame, schema)
    weights = validation_weights(
        valid_frame["LNMON"].to_numpy(), protocol.validation_month_shares,
    )
    completed = starting_round
    checkpoints = run_dir / "checkpoints" / name
    with tqdm(total=selected, initial=starting_round, desc=f"Training {name}", unit=" trees") as progress:
        while completed < selected:
            block = min(int(cfg.checkpoint.every), selected - completed)
            previous_models = fit_members(
                train_features, block, params, threads, previous_models,
                bool(cfg.runtime.parallel_members),
            )
            completed += block
            counts = boosted_rounds(previous_models)
            if any(count != completed for count in counts.values()):
                raise RuntimeError(f"Member round counts diverged: {counts}")
            scores = predict_members(
                previous_models, valid_features, threads,
                parallel=bool(cfg.runtime.parallel_members),
            )
            ensemble = average_probabilities(scores)
            record = {
                "round": completed,
                "valid_error": classification_error(valid_features.target, ensemble, weights, threshold),
                **{
                    f"{member}_error": classification_error(valid_features.target, value, weights, threshold)
                    for member, value in scores.items()
                },
            }
            history.append(record)
            save_checkpoint(checkpoints, previous_models, settings, history, completed)
            progress.update(block)
            progress.set_postfix_str(f"display-only valid error={record['valid_error']:.5f}")
            if plot is not None:
                plot.update(history, name)
    if previous_models is None:
        raise RuntimeError("Training finished without member models.")
    model_path = run_dir / f"{name}_model"
    save_models(model_path, previous_models)
    pd.DataFrame(history).to_csv(run_dir / f"{name}_history.csv", index=False)
    (run_dir / f"{name}_hyperparameters.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"{name}: final model at {model_path}; display-only valid 0/1 loss={history[-1]['valid_error']:.6f}")
    return {"experiment": name, "model_path": str(model_path),
            "display_only_valid_error": history[-1]["valid_error"]}


def run_training(cfg: DictConfig) -> Path:
    if (int(cfg.data.chunk_rows) < 1 or int(cfg.model.max_rounds) < 1
            or int(cfg.model.selection_every) < 1 or int(cfg.checkpoint.every) < 1):
        raise ValueError("CSV chunk size and boosting intervals must be positive.")
    if float(cfg.model.decision_threshold) != 0.5:
        raise ValueError("The primary score uses the fixed 0.5 threshold.")
    if cfg.runtime.live_plot not in ("auto", "on", "off"):
        raise ValueError("runtime.live_plot must be auto, on, or off.")
    source = Path(to_absolute_path(str(cfg.data.path)))
    frame = load_data(source, int(cfg.data.chunk_rows))
    source_hash = file_sha256(source)
    threads = int(cfg.runtime.threads_per_member) or max(1, (os.cpu_count() or 1) // 3)
    if threads < 1:
        raise ValueError("runtime.threads_per_member cannot be negative.")
    resume = None
    names = tuple(str(name) for name in cfg.runtime.models)
    if cfg.checkpoint.resume_from is not None:
        resume = load_checkpoint(Path(to_absolute_path(str(cfg.checkpoint.resume_from))), threads)
        names = (resume[0]["experiment"],)
    if not names or len(set(names)) != len(names) or any(name not in ("v1", "v2") for name in names):
        raise ValueError("runtime.models must contain v1, v2, or both without duplicates.")
    run_dir = create_run_directory(Path(to_absolute_path(str(cfg.output.base_dir))))
    backend = matplotlib.get_backend().lower()
    interactive = not any(word in backend for word in ("agg", "pdf", "svg", "ps", "cairo"))
    if cfg.runtime.live_plot == "on" and not interactive:
        raise ValueError("An interactive Matplotlib backend is required for live_plot=on.")
    plot = LivePlot(interactive and cfg.runtime.live_plot != "off")
    for name in names:
        train_protocol(cfg, name, frame, source, source_hash, run_dir, threads,
                       resume=resume, plot=plot)
        resume = None
    print(f"Ensemble run saved to: {run_dir}")
    return run_dir


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
