"""Fit two chronological GaussianNB models with 0/1-error monitoring."""

from __future__ import annotations

import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event

import hydra
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from . import WORKFLOW_VERSION
from .data import (
    Experiment, create_run_directory, experiment_from_config, file_sha256,
    fit_schema, load_data, transform_batch, validation_weights,
)
from .model import (
    HISTORY_COLUMNS, batch_slices, choose_var_smoothing, encode_to_memmap,
    load_checkpoint, misclassification, ordered_indices, predict_matrix_probabilities,
    save_checkpoint, save_final_model,
)


class LiveValidationPlot:
    """Update Matplotlib only on the main thread."""

    def __init__(self, names: tuple[str, ...]) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 5))
        self.axis.set(
            title="Validation misclassification rate during training",
            xlabel="Completed training batch",
            ylabel="Validation misclassification rate",
        )
        self.axis.grid(alpha=0.25)
        self.points = {name: ([], []) for name in names}
        self.lines = {name: self.axis.plot([], [], label=name)[0] for name in names}
        self.axis.legend()
        self.figure.show()

    def update(self, name: str, update: int, error: float) -> None:
        if not self.plt.fignum_exists(self.figure.number):
            return
        x_values, y_values = self.points[name]
        x_values.append(update)
        y_values.append(error)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def seed(self, name: str, history: list[dict]) -> None:
        for row in history:
            self.update(name, int(row["update"]), float(row["valid_error"]))

    def process_events(self) -> None:
        if self.plt.fignum_exists(self.figure.number):
            self.plt.pause(0.001)


def should_plot(mode: str) -> bool:
    if mode not in ("auto", "on", "off"):
        raise ValueError("runtime.live_plot must be auto, on, or off.")
    backend = matplotlib.get_backend().lower()
    interactive = backend not in ("agg", "pdf", "svg", "ps", "cairo", "template")
    if mode == "on" and not interactive:
        raise ValueError("runtime.live_plot=on requires an interactive Matplotlib backend.")
    return mode != "off" and interactive


def _settings(
    experiment: Experiment, frame: pd.DataFrame, source: Path, source_hash: str,
    cfg: DictConfig, selected: float, tuning_results: list[dict],
) -> dict:
    train_rows = int(frame["LNMON"].isin(experiment.train_months).sum())
    valid_rows = int(frame["LNMON"].isin(experiment.valid_months).sum())
    return {
        "workflow_version": WORKFLOW_VERSION,
        "experiment": experiment.name,
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "train_months": list(experiment.train_months),
        "validation_months": list(experiment.valid_months),
        "validation_month_shares": experiment.month_shares,
        "train_rows": train_rows,
        "validation_rows": valid_rows,
        "feature_columns": [name for name in frame.columns if name != "TARGET"],
        "hyperparameters": {
            "var_smoothing": selected,
            "var_smoothing_candidates": [float(value) for value in cfg.model.var_smoothing_candidates],
            "decision_threshold": float(cfg.model.decision_threshold),
            "batch_rows": int(cfg.training.batch_rows),
            "tuning_holdout_months": int(cfg.training.tuning_holdout_months),
        },
        "tuning_holdout_months": list(experiment.train_months[-int(cfg.training.tuning_holdout_months):]),
        "tuning_results": tuning_results,
        "history_metric": "misclassification_rate",
        "history_is_weighted": experiment.month_shares is not None,
        "checkpoint_every": int(cfg.checkpoint.every),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "hydra": hydra.__version__,
        },
    }


def _check_resume(previous: dict, current: dict) -> None:
    keys = (
        "workflow_version", "experiment", "source", "source_bytes", "source_sha256",
        "train_months", "validation_months", "validation_month_shares", "train_rows",
        "validation_rows", "feature_columns", "hyperparameters", "history_metric",
    )
    for key in keys:
        if previous[key] != json.loads(json.dumps(current[key])):
            raise ValueError(f"Checkpoint setting differs from this run: {key}")
    if previous["versions"]["scikit_learn"] != sklearn.__version__:
        raise ValueError("Resume requires the scikit-learn version used for the checkpoint.")


def train_experiment(
    experiment: Experiment, frame: pd.DataFrame, target: np.ndarray,
    months: np.ndarray, source: Path, source_hash: str, cfg: DictConfig,
    run_dir: Path, position: int, events: Queue,
    resume_bundle: tuple[dict, GaussianNB, dict, list[dict]] | None,
    stop_event: Event,
) -> dict:
    train_indices = ordered_indices(months, experiment.train_months)
    valid_indices = ordered_indices(months, experiment.valid_months)
    if np.unique(target[train_indices]).size != 2 or not len(valid_indices):
        raise ValueError(f"{experiment.name} needs both training classes and validation rows.")
    batch_rows = int(cfg.training.batch_rows)
    total_updates = math.ceil(len(train_indices) / batch_rows)
    weights = validation_weights(months[valid_indices], experiment.month_shares)
    if resume_bundle is None:
        selected, tuning_results = choose_var_smoothing(
            frame, months, target, experiment.train_months,
            [float(value) for value in cfg.model.var_smoothing_candidates],
            int(cfg.training.tuning_holdout_months), batch_rows,
            float(cfg.model.decision_threshold), experiment.name,
        )
        schema = fit_schema(frame, train_indices, experiment.name)
        model = GaussianNB(var_smoothing=selected)
        history: list[dict] = []
        settings = _settings(experiment, frame, source, source_hash, cfg, selected, tuning_results)
    else:
        previous, model, schema, history = resume_bundle
        settings = _settings(
            experiment, frame, source, source_hash, cfg,
            float(previous["hyperparameters"]["var_smoothing"]), previous["tuning_results"],
        )
        _check_resume(previous, settings)
        if schema["feature_columns"] != settings["feature_columns"]:
            raise ValueError("Checkpoint schema differs from the source columns.")
        if len(history) > total_updates or (
            history and history[-1]["train_rows"] != min(len(train_indices), len(history) * batch_rows)
        ):
            raise ValueError("Checkpoint training cursor is incompatible with the batch size.")
    checkpoint_root = run_dir / "checkpoints" / experiment.name
    batches = list(batch_slices(train_indices, batch_rows))
    with tempfile.TemporaryDirectory(prefix=f".{experiment.name}_validation_", dir=run_dir) as temporary:
        validation_matrix = encode_to_memmap(
            frame, valid_indices, schema, batch_rows,
            Path(temporary) / "validation.npy", experiment.name,
        )
        with tqdm(total=total_updates, initial=len(history), desc=f"Training {experiment.name}",
                  unit=" updates", position=position, leave=True) as progress:
            for batch_number in range(len(history), total_updates):
                if stop_event.is_set():
                    break
                indices = batches[batch_number]
                features = transform_batch(frame.iloc[indices], schema)
                model.partial_fit(features, target[indices],
                                  classes=np.array([0, 1]) if batch_number == 0 else None)
                probability = predict_matrix_probabilities(model, validation_matrix, batch_rows)
                error = misclassification(
                    target[valid_indices], probability, weights,
                    float(cfg.model.decision_threshold),
                )
                update = batch_number + 1
                history.append({
                    "update": update,
                    "train_rows": min(update * batch_rows, len(train_indices)),
                    "valid_error": error,
                })
                progress.update(1)
                progress.set_postfix_str(f"valid error={error:.5f}", refresh=False)
                if update == 1 or update % int(cfg.runtime.plot_every) == 0 or update == total_updates:
                    events.put((experiment.name, update, error))
                if update % int(cfg.checkpoint.every) == 0 or update == total_updates:
                    save_checkpoint(checkpoint_root, model, schema, settings, history)
            if stop_event.is_set() and history and not (checkpoint_root / f"update_{len(history):06d}").exists():
                save_checkpoint(checkpoint_root, model, schema, settings, history)
    if len(history) != total_updates:
        return {"experiment": experiment.name, "interrupted": True, "updates": len(history)}
    save_final_model(run_dir / f"{experiment.name}_model.pkl", model, schema)
    pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(
        run_dir / f"{experiment.name}_history.csv", index=False
    )
    (run_dir / f"{experiment.name}_hyperparameters.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "experiment": experiment.name,
        "interrupted": False,
        "updates": total_updates,
        "valid_error": history[-1]["valid_error"],
        "var_smoothing": settings["hyperparameters"]["var_smoothing"],
    }


def run_training(cfg: DictConfig) -> Path:
    if (int(cfg.data.chunk_rows) < 1 or int(cfg.training.batch_rows) < 1
            or int(cfg.checkpoint.every) < 1 or int(cfg.runtime.plot_every) < 1
            or not 0 < float(cfg.model.decision_threshold) < 1):
        raise ValueError("Chunk, batch, checkpoint, plot, and threshold settings are invalid.")
    candidates = [float(value) for value in cfg.model.var_smoothing_candidates]
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("var_smoothing candidates must be positive and finite.")
    names = tuple(str(name) for name in cfg.runtime.models)
    if not names or len(set(names)) != len(names) or any(name not in cfg.protocols for name in names):
        raise ValueError("runtime.models must contain distinct configured protocols.")
    experiments = tuple(experiment_from_config(name, cfg) for name in names)
    source = Path(to_absolute_path(str(cfg.data.path)))
    output_base = Path(to_absolute_path(str(cfg.output.base_dir)))
    resume_bundle = None
    if cfg.checkpoint.resume_from is not None:
        if len(experiments) != 1:
            raise ValueError("Select one runtime model when resuming a checkpoint.")
        resume_path = Path(to_absolute_path(str(cfg.checkpoint.resume_from)))
        resume_bundle = load_checkpoint(resume_path)
        if resume_bundle[0]["experiment"] != names[0]:
            raise ValueError("Checkpoint protocol does not match runtime.models.")
    frame = load_data(source, int(cfg.data.chunk_rows))
    source_hash = file_sha256(source)
    target = frame["TARGET"].to_numpy(dtype="int8")
    months = frame["LNMON"].to_numpy(dtype="int32")
    run_dir = create_run_directory(output_base)
    print(f"Run directory: {run_dir}")
    plot = LiveValidationPlot(names) if should_plot(str(cfg.runtime.live_plot)) else None
    if plot is not None and resume_bundle is not None:
        plot.seed(names[0], resume_bundle[3])
    if plot is None:
        print("Live plotting is unavailable or disabled; valid.py will plot saved history.")
    workers = min(len(experiments), os.cpu_count() or 1) if bool(cfg.runtime.parallel_models) else 1
    events: Queue = Queue()
    stop_event = Event()
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                train_experiment, experiment, frame, target, months, source, source_hash,
                cfg, run_dir, position, events,
                resume_bundle if resume_bundle is not None else None, stop_event,
            )
            for position, experiment in enumerate(experiments, start=1)
        ]
        try:
            while not all(future.done() for future in futures) or not events.empty():
                try:
                    name, update, error = events.get(timeout=0.2)
                    if plot is not None:
                        plot.update(name, update, error)
                except Empty:
                    if plot is not None:
                        plot.process_events()
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception()
            reports = [future.result() for future in futures]
        except KeyboardInterrupt:
            stop_event.set()
            reports = [future.result() for future in futures]
    for report in sorted(reports, key=lambda row: row["experiment"]):
        if report["interrupted"]:
            print(f"{report['experiment']}: interrupted after update {report['updates']}; resume from a checkpoint")
        else:
            print(
                f"{report['experiment']}: validation misclassification={report['valid_error']:.6f}, "
                f"var_smoothing={report['var_smoothing']:g}"
            )
    print(f"Results saved to: {run_dir}")
    return run_dir


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
