"""Train two chronological Elastic Net logistic models with full SAGA snapshots."""

from __future__ import annotations

import json
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
import scipy
import sklearn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from . import WORKFLOW_VERSION
from .data import (
    Experiment, create_run_directory, experiment_from_config, file_sha256,
    fit_schema, load_data, transform_to_memmap, validation_weights,
)
from .saga import (
    check_state, initial_state, load_checkpoint, predict_probability,
    run_epoch, save_checkpoint, save_final_model,
)


class LiveValidationPlot:
    """Draw training events on the main GUI thread."""

    def __init__(self, names: tuple[str, ...]) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 5))
        self.axis.set(
            title="Validation misclassification rate during training",
            xlabel="Completed SAGA epoch", ylabel="Validation misclassification rate",
        )
        self.axis.grid(alpha=0.25)
        self.points = {name: ([], []) for name in names}
        self.lines = {name: self.axis.plot([], [], label=name)[0] for name in names}
        self.axis.legend()
        self.figure.show()

    def update(self, name: str, epoch: int, error: float) -> None:
        if not self.plt.fignum_exists(self.figure.number):
            return
        x_values, y_values = self.points[name]
        x_values.append(epoch)
        y_values.append(error)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def process_events(self) -> None:
        if self.plt.fignum_exists(self.figure.number):
            self.plt.pause(0.001)


def available_cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def should_plot(mode: str) -> bool:
    if mode not in ("auto", "on", "off"):
        raise ValueError("runtime.live_plot must be auto, on, or off.")
    backend = matplotlib.get_backend().lower()
    interactive = not any(token in backend for token in ("agg", "pdf", "svg", "ps", "cairo"))
    if mode == "on" and not interactive:
        raise ValueError("runtime.live_plot=on requires an interactive Matplotlib backend.")
    return mode == "on" or (mode == "auto" and interactive)


def _settings(
    experiment: Experiment, schema: dict, source: Path, source_sha256: str,
    frame: pd.DataFrame, cfg: DictConfig,
) -> dict:
    months = frame["LNMON"].to_numpy()
    return {
        "workflow_version": WORKFLOW_VERSION,
        "experiment": experiment.name,
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha256,
        "train_months": list(experiment.train_months),
        "validation_months": list(experiment.valid_months),
        "validation_month_shares": experiment.month_shares,
        "train_rows": int(np.isin(months, experiment.train_months).sum()),
        "validation_rows": int(np.isin(months, experiment.valid_months).sum()),
        "schema": schema,
        "hyperparameters": OmegaConf.to_container(cfg.model, resolve=True),
        "classification_threshold": 0.5,
        "history_metric": "misclassification_rate",
        "fit_objective": "elastic_net_penalized_log_loss",
        "final_model_selection": "last_completed_epoch",
        "random_schedule": "SeedSequence([random_state, epoch + 1])",
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def _worker(
    experiment: Experiment,
    frame: pd.DataFrame,
    source: Path,
    source_sha256: str,
    cfg: DictConfig,
    run_dir: Path,
    resume_path: Path | None,
    events: Queue,
    stop_event: Event,
    position: int,
) -> dict:
    months = frame["LNMON"].to_numpy()
    train_indices = np.flatnonzero(np.isin(months, experiment.train_months))
    valid_indices = np.flatnonzero(np.isin(months, experiment.valid_months))
    if not len(train_indices) or not len(valid_indices):
        raise ValueError(f"{experiment.name} has no training or validation rows.")
    if set(frame["TARGET"].iloc[train_indices].unique()) != {0, 1}:
        raise ValueError(f"{experiment.name} training rows need both target classes.")
    previous_settings = None
    state = None
    history: list[dict] = []
    if resume_path is not None:
        previous_settings, state, history = load_checkpoint(resume_path)
        schema = previous_settings["schema"]
    else:
        schema = fit_schema(frame, train_indices, position=position, label=experiment.name)
    settings = _settings(experiment, schema, source, source_sha256, frame, cfg)
    if previous_settings is not None and previous_settings != json.loads(json.dumps(settings)):
        raise ValueError("Checkpoint settings, source, schema, or package versions differ from this run.")
    weights = validation_weights(months[valid_indices], experiment.month_shares)
    y_train = frame["TARGET"].iloc[train_indices].to_numpy(dtype="float32")
    y_valid = frame["TARGET"].iloc[valid_indices].to_numpy(dtype="int8")
    checkpoint_root = run_dir / "checkpoints" / experiment.name
    params = settings["hyperparameters"]
    with tempfile.TemporaryDirectory(prefix=f".{experiment.name}_features_", dir=run_dir) as workspace:
        workspace_path = Path(workspace)
        x_train, _ = transform_to_memmap(
            frame, train_indices, schema, workspace_path / "train.npy",
            show_progress=True, position=position, label=f"{experiment.name} train",
        )
        x_valid, _ = transform_to_memmap(
            frame, valid_indices, schema, workspace_path / "valid.npy",
            show_progress=True, position=position, label=f"{experiment.name} valid",
        )
        if state is None:
            state = initial_state(x_train, float(params["C"]), float(params["l1_ratio"]), int(params["random_state"]))
        else:
            check_state(state, x_train, int(params["random_state"]))
        max_epochs = int(params["max_epochs"])
        converged = bool(history and history[-1]["relative_coef_change"] <= float(params["tol"]))
        with tqdm(
            total=max_epochs, initial=state.epoch, desc=f"{experiment.name} SAGA",
            unit=" epochs", position=position, leave=True,
        ) as progress:
            while state.epoch < max_epochs and not converged and not stop_event.is_set():
                change = run_epoch(state, x_train, y_train, float(params["C"]), float(params["l1_ratio"]))
                probability = predict_probability(x_valid, state.coef, state.intercept)
                error = float(np.average((probability > 0.5) != y_valid, weights=weights))
                history.append({"epoch": state.epoch, "valid_error": error, "relative_coef_change": change})
                progress.update(1)
                progress.set_postfix_str(f"valid error={error:.5f}", refresh=False)
                if state.epoch == 1 or state.epoch % int(cfg.runtime.plot_every) == 0 or state.epoch == max_epochs:
                    events.put((experiment.name, state.epoch, error))
                converged = change <= float(params["tol"])
                if state.epoch % int(cfg.checkpoint.every) == 0 or converged or state.epoch == max_epochs:
                    save_checkpoint(checkpoint_root, state, settings, history)
                if converged:
                    break
        if stop_event.is_set():
            if state.epoch and not (checkpoint_root / f"epoch_{state.epoch:06d}").exists():
                save_checkpoint(checkpoint_root, state, settings, history)
            return {"experiment": experiment.name, "interrupted": True, "epoch": state.epoch}
        if not history:
            raise RuntimeError("Training did not complete any SAGA epoch.")
        save_final_model(run_dir / f"{experiment.name}_model.pkl", schema, state)
        pd.DataFrame(history).to_csv(run_dir / f"{experiment.name}_history.csv", index=False)
        final_settings = {
            **settings,
            "actual_epochs": state.epoch,
            "converged": converged,
            "final_valid_error": history[-1]["valid_error"],
            "best_validation_epoch": int(min(history, key=lambda row: row["valid_error"])["epoch"]),
        }
        (run_dir / f"{experiment.name}_hyperparameters.json").write_text(
            json.dumps(final_settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {
            "experiment": experiment.name, "interrupted": False,
            "epoch": state.epoch, "valid_error": history[-1]["valid_error"],
            "converged": converged,
        }


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    if str(cfg.model.solver) != "saga" or str(cfg.model.penalty) != "elasticnet":
        raise ValueError("This workflow requires solver=saga and penalty=elasticnet.")
    if not 0 <= float(cfg.model.l1_ratio) <= 1 or float(cfg.model.C) <= 0:
        raise ValueError("model.l1_ratio must be in [0, 1] and model.C positive.")
    if int(cfg.model.max_epochs) < 1 or float(cfg.model.tol) < 0 or int(cfg.model.random_state) < 0:
        raise ValueError("max_epochs must be positive, tol nonnegative, and random_state nonnegative.")
    if int(cfg.checkpoint.every) < 1 or int(cfg.runtime.plot_every) < 1 or int(cfg.runtime.training_threads) < 0:
        raise ValueError("Checkpoint, plot interval, and training thread settings are invalid.")
    names = tuple(str(name) for name in cfg.runtime.models)
    if not names or len(set(names)) != len(names):
        raise ValueError("runtime.models must contain unique protocol names.")
    resume_path = Path(to_absolute_path(str(cfg.checkpoint.resume_from))) if cfg.checkpoint.resume_from else None
    if resume_path is not None:
        previous_settings, _, _ = load_checkpoint(resume_path)
        names = (previous_settings["experiment"],)
    experiments = tuple(experiment_from_config(name, cfg) for name in names)
    plot = LiveValidationPlot(names) if should_plot(str(cfg.runtime.live_plot)) else None
    source = Path(to_absolute_path(str(cfg.data.path)))
    source_sha256 = file_sha256(source)
    frame = load_data(source, int(cfg.data.chunk_rows))
    run_dir = create_run_directory(Path(to_absolute_path(str(cfg.output.base_dir))))
    print(f"Elastic Net run directory: {run_dir}")
    workers = min(len(experiments), int(cfg.runtime.training_threads) or available_cpus()) if cfg.runtime.parallel_models else 1
    events: Queue = Queue()
    stop_event = Event()
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _worker, experiment, frame, source, source_sha256, cfg, run_dir,
                resume_path, events, stop_event, index + 1,
            )
            for index, experiment in enumerate(experiments)
        ]
        interrupted = False
        while any(not future.done() for future in futures) or not events.empty():
            if any(future.done() and future.exception() is not None for future in futures):
                stop_event.set()
            try:
                name, epoch, error = events.get(timeout=0.2)
                if plot is not None:
                    plot.update(name, epoch, error)
                if plot is not None:
                    plot.process_events()
            except Empty:
                if plot is not None:
                    plot.process_events()
            except KeyboardInterrupt:
                interrupted = True
                stop_event.set()
        results = [future.result() for future in futures]
    for result in results:
        if result["interrupted"]:
            print(f"{result['experiment']}: interrupted after epoch {result['epoch']}; checkpoint saved")
        else:
            print(
                f"{result['experiment']}: epoch={result['epoch']}, "
                f"final validation misclassification={result['valid_error']:.6f}, "
                f"converged={result['converged']}"
            )
    if interrupted:
        print("Training interrupted. Resume from a saved checkpoint directory.")


if __name__ == "__main__":
    main()
