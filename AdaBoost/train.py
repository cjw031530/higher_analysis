"""Train two chronological AdaBoost-SAMME models with resumable checkpoints."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import Empty

import hydra
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import tqdm as tqdm_package
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from . import WORKFLOW_VERSION
from .data import (
    Experiment, create_run_directory, encode_features, experiment_from_config,
    feature_schema, file_sha256, load_data, validation_weights,
)
from .samme import (
    SAMMEModel, boost_one, initial_snapshot, load_checkpoint,
    save_checkpoint, save_final_model,
)


class LiveValidationPlot:
    """Render worker-reported validation errors on the main GUI thread."""

    def __init__(self, names: tuple[str, ...]) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 5))
        self.axis.set(
            title="Validation misclassification rate during training",
            xlabel="Boosting iteration",
            ylabel="Validation misclassification rate",
        )
        self.axis.grid(alpha=0.25)
        self.points = {name: ([], []) for name in names}
        self.lines = {name: self.axis.plot([], [], label=name)[0] for name in names}
        self.axis.legend()
        self.figure.show()

    def update(self, name: str, iteration: int, error: float) -> None:
        if not self.plt.fignum_exists(self.figure.number):
            return
        x_values, y_values = self.points[name]
        x_values.append(iteration)
        y_values.append(error)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def seed(self, name: str, history: list[float]) -> None:
        x_values, y_values = self.points[name]
        x_values.extend(range(1, len(history) + 1))
        y_values.extend(history)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()

    def process_events(self) -> None:
        if self.plt.fignum_exists(self.figure.number):
            self.plt.pause(0.001)


def _check_resume(state: dict, settings: dict) -> None:
    comparable = (
        "experiment", "workflow_version", "source", "source_bytes", "source_sha256",
        "train_months", "validation_months", "validation_month_shares", "train_rows",
        "validation_rows", "schema", "hyperparameters", "history_metric",
        "classification_threshold", "final_model_selection",
    )
    for key in comparable:
        if state.get(key) != json.loads(json.dumps(settings[key])):
            raise ValueError(f"Checkpoint setting differs from this run: {key}")
    if state["versions"]["scikit_learn"] != sklearn.__version__:
        raise ValueError("Resume requires the scikit-learn version used for the checkpoint.")


def train_experiment_worker(
    experiment: Experiment,
    matrix_path: Path,
    target: np.ndarray,
    months: np.ndarray,
    settings: dict,
    run_dir: Path,
    checkpoint_every: int,
    event_queue,
    resume_path: Path | None,
) -> dict:
    """Fit one model in a separate process from the read-only encoded matrix."""

    matrix = np.load(matrix_path, mmap_mode="r")
    train_mask = np.isin(months, experiment.train_months)
    valid_mask = np.isin(months, experiment.valid_months)
    x_train = np.ascontiguousarray(matrix[train_mask], dtype=np.float32)
    x_valid = np.ascontiguousarray(matrix[valid_mask], dtype=np.float32)
    y_train = target[train_mask]
    y_valid = target[valid_mask]
    weights = validation_weights(pd.Series(months[valid_mask]), experiment.month_shares)
    params = settings["hyperparameters"]
    if resume_path is None:
        snapshot = initial_snapshot(SAMMEModel(**params), len(y_train), len(y_valid))
    else:
        previous, snapshot = load_checkpoint(resume_path)
        _check_resume(previous, settings)
        if snapshot.sample_weight.shape != (len(y_train),) or snapshot.valid_margin.shape != (len(y_valid),):
            raise ValueError("Checkpoint arrays do not match the current split.")
    checkpoint_root = run_dir / "checkpoints" / experiment.name
    while snapshot.model.fitted_iterations < snapshot.model.n_estimators and snapshot.stopping_reason is None:
        error = boost_one(snapshot, x_train, y_train, x_valid, y_valid, weights)
        if error is None:
            break
        iteration = snapshot.model.fitted_iterations
        event_queue.put((experiment.name, iteration, error))
        if iteration % checkpoint_every == 0 or iteration == snapshot.model.n_estimators or snapshot.stopping_reason is not None:
            save_checkpoint(checkpoint_root, snapshot, settings)
    if snapshot.model.fitted_iterations == 0:
        raise RuntimeError("Training ended without a fitted AdaBoost tree.")
    if snapshot.stopping_reason is not None:
        terminal = checkpoint_root / f"round_{snapshot.model.fitted_iterations:06d}"
        if not terminal.exists():
            save_checkpoint(checkpoint_root, snapshot, settings)
    save_final_model(run_dir / f"{experiment.name}_model.pkl", snapshot.model)
    pd.DataFrame({
        "iteration": np.arange(1, len(snapshot.history) + 1),
        "valid_error": snapshot.history,
    }).to_csv(run_dir / f"{experiment.name}_history.csv", index=False)
    final_settings = {
        **settings,
        "actual_estimators": snapshot.model.fitted_iterations,
        "stopping_reason": snapshot.stopping_reason,
        "final_valid_error": snapshot.history[-1],
        "best_validation_iteration": int(np.argmin(snapshot.history)) + 1,
        "best_validation_error": float(np.min(snapshot.history)),
    }
    (run_dir / f"{experiment.name}_hyperparameters.json").write_text(
        json.dumps(final_settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "experiment": experiment.name,
        "actual_estimators": snapshot.model.fitted_iterations,
        "final_valid_error": snapshot.history[-1],
        "stopping_reason": snapshot.stopping_reason,
    }


def _settings(
    experiment: Experiment,
    schema: dict,
    frame: pd.DataFrame,
    source: Path,
    source_hash: str,
    model_params: dict,
) -> dict:
    train_mask = frame["LNMON"].isin(experiment.train_months).to_numpy()
    valid_mask = frame["LNMON"].isin(experiment.valid_months).to_numpy()
    if not train_mask.any() or not valid_mask.any():
        raise ValueError(f"{experiment.name} has an empty train or validation split.")
    if set(frame.loc[train_mask, "LNMON"]) != set(experiment.train_months):
        raise ValueError(f"{experiment.name} is missing one or more training months.")
    if set(frame.loc[valid_mask, "LNMON"]) != set(experiment.valid_months):
        raise ValueError(f"{experiment.name} is missing one or more validation months.")
    if set(frame.loc[train_mask, "TARGET"]) != {0, 1}:
        raise ValueError(f"{experiment.name} training requires both TARGET classes.")
    validation_weights(frame.loc[valid_mask, "LNMON"], experiment.month_shares)
    unseen = {}
    for name, levels in schema["categorical_levels"].items():
        values = frame.loc[valid_mask, name].astype("string")
        unseen[name] = int((values.notna() & ~values.isin(levels)).sum())
    return {
        "experiment": experiment.name,
        "workflow_version": WORKFLOW_VERSION,
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "train_months": list(experiment.train_months),
        "validation_months": list(experiment.valid_months),
        "validation_month_shares": experiment.month_shares,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "schema": schema,
        "unseen_validation_categories_mapped_to_missing": unseen,
        "hyperparameters": model_params,
        "history_metric": "misclassification_rate",
        "classification_threshold": 0.5,
        "final_model_selection": "last_fitted_iteration",
        "fit_algorithm": "binary_samme_weighted_weak_learner_error",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tqdm": tqdm_package.__version__,
            "matplotlib": matplotlib.__version__,
            "hydra": hydra.__version__,
        },
    }


def run_training(cfg: DictConfig) -> Path:
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    if (
        int(cfg.data.chunk_rows) < 1
        or int(cfg.model.max_depth) < 1
        or int(cfg.model.n_estimators) < 1
        or float(cfg.model.learning_rate) <= 0
        or int(cfg.checkpoint.every) < 1
        or int(cfg.runtime.plot_every) < 1
    ):
        raise ValueError("Chunk size, depth, tree count, learning rate, and intervals must be positive.")
    if cfg.runtime.live_plot not in ("auto", "on", "off"):
        raise ValueError("runtime.live_plot must be auto, on, or off.")
    backend = matplotlib.get_backend().lower()
    interactive = backend not in {"agg", "pdf", "ps", "svg", "template", "cairo"} and "inline" not in backend
    if cfg.runtime.live_plot == "on" and not interactive:
        raise RuntimeError(f"Matplotlib backend {backend!r} cannot show a live plot.")
    show_plot = interactive and cfg.runtime.live_plot != "off"
    source = Path(to_absolute_path(str(cfg.data.path)))
    output_base = Path(to_absolute_path(str(cfg.output.base_dir)))
    resume_path = None
    resume_history: list[float] = []
    if cfg.checkpoint.resume_from is not None:
        resume_path = Path(to_absolute_path(str(cfg.checkpoint.resume_from)))
        resume_state, resume_snapshot = load_checkpoint(resume_path)
        names = (str(resume_state["experiment"]),)
        resume_history = resume_snapshot.history
    else:
        names = tuple(str(name) for name in cfg.runtime.models)
    if not names or len(set(names)) != len(names) or any(name not in cfg.protocols for name in names):
        raise ValueError("runtime.models must contain distinct configured protocol names.")
    experiments = tuple(experiment_from_config(name, cfg) for name in names)
    frame = load_data(source, int(cfg.data.chunk_rows))
    source_hash = file_sha256(source)
    run_dir = create_run_directory(output_base)
    print(f"Run directory: {run_dir}")
    plot = LiveValidationPlot(names) if show_plot else None
    if plot is not None and resume_history:
        plot.seed(names[0], resume_history)
    if not show_plot:
        print("Live plotting is unavailable or disabled; valid.py will plot saved history.")
    # The matrix is written once per protocol and shared read-only by its worker.
    # A temporary directory ensures these large arrays are absent from final artifacts.
    with tempfile.TemporaryDirectory(prefix=".matrix_", dir=run_dir) as temporary:
        matrix_dir = Path(temporary)
        jobs = []
        for experiment in experiments:
            train_mask = frame["LNMON"].isin(experiment.train_months).to_numpy()
            schema = feature_schema(frame, train_mask)
            settings = _settings(experiment, schema, frame, source, source_hash, model_params)
            matrix_path = matrix_dir / f"{experiment.name}_features.npy"
            encode_features(frame, schema, output_path=matrix_path, show_progress=True)
            jobs.append((experiment, matrix_path, settings))
        target = frame["TARGET"].to_numpy(dtype="int8", copy=True)
        months = frame["LNMON"].to_numpy(dtype="int32", copy=True)
        del frame
        # Limit threaded numerical libraries in spawned workers. The tree fit itself
        # is sequential; independent protocols run in separate processes.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = "1"
        context = mp.get_context("spawn")
        workers = len(experiments) if bool(cfg.runtime.parallel_models) else 1
        with context.Manager() as manager, ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            events = manager.Queue()
            futures = [
                pool.submit(
                    train_experiment_worker,
                    experiment, matrix_path, target, months, settings, run_dir,
                    int(cfg.checkpoint.every), events, resume_path,
                )
                for experiment, matrix_path, settings in jobs
            ]
            with ExitStack() as stack:
                bars = {
                    experiment.name: stack.enter_context(tqdm(
                        total=int(cfg.model.n_estimators),
                        initial=len(resume_history) if resume_path is not None else 0,
                        desc=f"Training {experiment.name}",
                        unit=" trees", position=position, leave=True,
                    ))
                    for position, experiment in enumerate(experiments, start=1)
                }
                while True:
                    try:
                        name, iteration, error = events.get(timeout=0.2)
                        bar = bars[name]
                        bar.update(iteration - bar.n)
                        bar.set_postfix_str(f"valid error={error:.5f}", refresh=False)
                        if plot is not None and (iteration == 1 or iteration % int(cfg.runtime.plot_every) == 0 or iteration == int(cfg.model.n_estimators)):
                            plot.update(name, iteration, error)
                    except Empty:
                        if plot is not None:
                            plot.process_events()
                        if all(future.done() for future in futures):
                            break
                reports = [future.result() for future in futures]
    for report in sorted(reports, key=lambda item: item["experiment"]):
        print(
            f"{report['experiment']}: final valid misclassification rate="
            f"{report['final_valid_error']:.6f} after {report['actual_estimators']} trees"
        )
        if report["stopping_reason"] is not None:
            print(f"{report['experiment']}: AdaBoost stopped: {report['stopping_reason']}")
    print(f"Results saved to: {run_dir}")
    return run_dir


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
