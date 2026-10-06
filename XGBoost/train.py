"""Train two time-based XGBoost classifiers and record validation curves."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import hydra
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import tqdm as tqdm_package
import xgboost as xgb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


ALL_MONTHS = (
    202306, 202307, 202308, 202309, 202310, 202311,
    202312, 202401, 202402, 202403, 202404, 202405,
)
WORKFLOW_VERSION = "3.1.0"


@dataclass(frozen=True)
class Experiment:
    name: str
    train_months: tuple[int, ...]
    valid_months: tuple[int, ...]
    month_shares: dict[int, float] | None


def experiment_from_config(name: str, cfg: DictConfig) -> Experiment:
    protocol = cfg.protocols[name]
    shares = protocol.validation_month_shares
    parsed_shares = None if shares is None else {int(k): float(v) for k, v in shares.items()}
    return Experiment(
        name,
        tuple(int(month) for month in protocol.train_months),
        tuple(int(month) for month in protocol.validation_months),
        parsed_shares,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointWriter:
    """Write a complete, atomically published checkpoint at fixed rounds."""

    def __init__(
        self, root: Path, settings: dict, prefix_history: list[float]
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.prefix_history = prefix_history

    def save(self, booster: xgb.Booster, iteration: int, new_history: list[float]) -> Path:
        history = self.prefix_history + list(new_history)
        if len(history) != iteration:
            raise RuntimeError("Checkpoint history does not match the boosted round count.")
        destination = self.root / f"round_{iteration:06d}"
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=self.root))
        try:
            with (temporary / "model.pkl").open("wb") as handle:
                pickle.dump(booster, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pd.DataFrame({
                "iteration": np.arange(1, iteration + 1),
                "valid_error": history,
            }).to_csv(temporary / "history.csv", index=False)
            (temporary / "state.json").write_text(
                json.dumps({**self.settings, "checkpoint_iteration": iteration}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination


def load_checkpoint(path: Path) -> tuple[dict, list[float], xgb.Booster]:
    directory = path if path.is_dir() else path.parent
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    history_frame = pd.read_csv(directory / "history.csv")
    if list(history_frame.columns) != ["iteration", "valid_error"]:
        raise ValueError("Checkpoint history has unexpected columns.")
    iteration = int(state["checkpoint_iteration"])
    if history_frame["iteration"].tolist() != list(range(1, iteration + 1)):
        raise ValueError("Checkpoint history is incomplete.")
    if state["workflow_version"] != WORKFLOW_VERSION:
        raise ValueError("The checkpoint workflow version is incompatible.")
    if state["versions"]["xgboost"] != xgb.__version__:
        raise ValueError("Resume requires the XGBoost version used for the checkpoint.")
    with (directory / "model.pkl").open("rb") as handle:
        booster = pickle.load(handle)
    if booster.num_boosted_rounds() != iteration:
        raise ValueError("Checkpoint model rounds do not match its state.")
    return state, history_frame["valid_error"].tolist(), booster


class TrainingProgress(xgb.callback.TrainingCallback):
    """Update tqdm and send periodic validation error to the plotting thread."""

    def __init__(
        self,
        progress: tqdm,
        experiment_name: str,
        total_rounds: int,
        starting_round: int,
        plot_every: int,
        plot_events: Queue | None,
        checkpoint_every: int,
        checkpoint_writer: CheckpointWriter,
    ) -> None:
        self.progress = progress
        self.experiment_name = experiment_name
        self.total_rounds = total_rounds
        self.starting_round = starting_round
        self.plot_every = plot_every
        self.plot_events = plot_events
        self.checkpoint_every = checkpoint_every
        self.checkpoint_writer = checkpoint_writer

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict
    ) -> bool:
        iteration = self.starting_round + epoch + 1
        self.progress.update(iteration - self.progress.n)
        history = evals_log["validation_0"]["error"]
        if iteration % self.checkpoint_every == 0 or iteration == self.total_rounds:
            self.checkpoint_writer.save(model, iteration, history)
        if epoch == 0 or iteration % self.plot_every == 0 or iteration == self.total_rounds:
            error = float(history[-1])
            self.progress.set_postfix_str(f"valid error={error:.5f}", refresh=False)
            if self.plot_events is not None:
                self.plot_events.put((self.experiment_name, iteration, error))
        return False


class LiveValidationPlot:
    """Draw concurrent training curves from events on the main GUI thread."""

    def __init__(self, experiment_names: tuple[str, ...]) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 5))
        self.axis.set_title("Validation misclassification rate during training")
        self.axis.set_xlabel("Boosting iteration")
        self.axis.set_ylabel("Validation misclassification rate")
        self.axis.grid(alpha=0.25)
        self.points = {name: ([], []) for name in experiment_names}
        self.lines = {
            name: self.axis.plot([], [], label=name)[0]
            for name in experiment_names
        }
        self.axis.legend()
        self.figure.show()

    def update(self, event: tuple[str, int, float]) -> None:
        if not self.plt.fignum_exists(self.figure.number):
            return
        name, iteration, error = event
        x_values, y_values = self.points[name]
        x_values.append(iteration)
        y_values.append(error)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def seed(self, name: str, history: list[float]) -> None:
        if not history:
            return
        x_values, y_values = self.points[name]
        x_values.extend(range(1, len(history) + 1))
        y_values.extend(history)
        self.lines[name].set_data(x_values, y_values)
        self.axis.relim()
        self.axis.autoscale_view()

    def process_events(self) -> None:
        if self.plt.fignum_exists(self.figure.number):
            self.plt.pause(0.001)


def load_data(path: Path, chunk_rows: int, require_all_months: bool = True) -> pd.DataFrame:
    """Read the source CSV with visible row progress and validate its schema."""

    if not path.is_file():
        raise FileNotFoundError(path)
    chunks: list[pd.DataFrame] = []
    with tqdm(desc="Reading CSV", unit=" rows", position=0) as progress:
        for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
            chunks.append(chunk)
            progress.update(len(chunk))
    if not chunks:
        raise ValueError("The source CSV is empty.")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    if not frame.columns.is_unique:
        raise ValueError("The source CSV has duplicate column names.")
    if not {"LNMON", "TARGET"}.issubset(frame.columns):
        raise ValueError("The source CSV must contain LNMON and TARGET.")
    if frame[["LNMON", "TARGET"]].isna().any(axis=None):
        raise ValueError("LNMON and TARGET must not be missing.")

    frame["LNMON"] = pd.to_numeric(frame["LNMON"], errors="raise")
    frame["TARGET"] = pd.to_numeric(frame["TARGET"], errors="raise")
    if not frame["LNMON"].isin(ALL_MONTHS).all():
        raise ValueError("LNMON contains a month outside 202306 through 202405.")
    if not frame["TARGET"].isin((0, 1)).all():
        raise ValueError("TARGET must contain only 0 and 1.")
    frame["LNMON"] = frame["LNMON"].astype("int32")
    frame["TARGET"] = frame["TARGET"].astype("int8")
    observed = set(frame["LNMON"].unique())
    if require_all_months and observed != set(ALL_MONTHS):
        missing = sorted(set(ALL_MONTHS) - observed)
        raise ValueError(f"The source CSV is missing required months: {missing}")
    if len(frame.columns) < 3:
        raise ValueError("The source CSV has no predictor columns.")
    numeric_columns = frame.select_dtypes(include="number").columns
    for name in tqdm(numeric_columns, desc="Compacting numeric columns", unit=" columns"):
        series = frame[name]
        if pd.api.types.is_integer_dtype(series):
            frame[name] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            observed_values = series.dropna().to_numpy(dtype="float64", copy=False)
            if len(observed_values) == 0:
                frame[name] = series.astype("float32")
            elif (
                np.isfinite(observed_values).all()
                and observed_values.min() >= -(2**31)
                and observed_values.max() <= 2**31 - 1
                and np.equal(observed_values, np.trunc(observed_values)).all()
            ):
                frame[name] = series.astype("Int32")
    return frame


def prepare_features(
    train: pd.DataFrame, valid: pd.DataFrame, categorical: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Use training-only category vocabularies for each validation fold."""

    x_train = train.drop(columns="TARGET").copy(deep=False)
    x_valid = valid.drop(columns="TARGET").copy(deep=False)
    unseen_counts: dict[str, int] = {}
    for name in categorical:
        train_values = x_train[name].astype("string")
        valid_values = x_valid[name].astype("string")
        categories = pd.Index(train_values.dropna().unique()).sort_values()
        category_type = pd.CategoricalDtype(categories=categories)
        x_train[name] = train_values.astype(category_type)
        unseen_mask = valid_values.notna() & ~valid_values.isin(categories)
        unseen_counts[name] = int(unseen_mask.sum())
        x_valid[name] = valid_values.mask(unseen_mask).astype(category_type)
    return x_train, x_valid, unseen_counts


def validation_weights(
    months: pd.Series, month_shares: dict[int, float] | None
) -> np.ndarray | None:
    """Give each requested month its exact total share in validation."""

    if month_shares is None:
        return None
    counts = months.value_counts()
    if set(counts.index) != set(month_shares) or not np.isclose(sum(month_shares.values()), 1.0):
        raise ValueError("Validation shares must cover all validation months and total one.")
    weight_by_month = {
        month: share / int(counts[month])
        for month, share in month_shares.items()
    }
    return months.map(weight_by_month).to_numpy(dtype="float64")


def create_run_directory(base: Path, prefix: str = "run") -> Path:
    """Create a new output directory without replacing an earlier experiment."""

    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        stem = f"{prefix}_{stamp}"
        run_dir = base / (stem if suffix == 0 else f"{stem}_{suffix}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique output directory.")


def train_experiment(
    experiment: Experiment,
    frame: pd.DataFrame,
    categorical: tuple[str, ...],
    run_dir: Path,
    model_params: dict,
    jobs_per_model: int,
    progress_position: int,
    plot_every: int,
    plot_events: Queue | None,
    source_path: Path,
    source_hash: str,
    checkpoint_every: int,
    resume_bundle: tuple[dict, list[float], xgb.Booster] | None,
) -> dict:
    """Fit one model, checkpoint it, and write its final portable artifacts."""

    train = frame.loc[frame["LNMON"].isin(experiment.train_months)]
    valid = frame.loc[frame["LNMON"].isin(experiment.valid_months)]
    x_train, x_valid, unseen_counts = prepare_features(train, valid, categorical)
    y_train = train["TARGET"].to_numpy()
    y_valid = valid["TARGET"].to_numpy()
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"{experiment.name} training requires both target classes.")
    weights = validation_weights(valid["LNMON"], experiment.month_shares)
    params = {**model_params, "n_jobs": jobs_per_model}
    total_rounds = int(params["n_estimators"])
    settings = {
        "experiment": experiment.name,
        "workflow_version": WORKFLOW_VERSION,
        "source": str(source_path.resolve()),
        "source_bytes": source_path.stat().st_size,
        "source_sha256": source_hash,
        "train_months": experiment.train_months,
        "validation_months": experiment.valid_months,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "feature_columns": list(x_train.columns),
        "categorical_features": categorical,
        "category_levels": {
            name: x_train[name].cat.categories.tolist() for name in categorical
        },
        "unseen_validation_categories_mapped_to_missing": unseen_counts,
        "hyperparameters": params,
        "validation_month_shares": experiment.month_shares,
        "history_metric": "error",
        "classification_threshold": 0.5,
        "history_is_weighted": weights is not None,
        "checkpoint_every": checkpoint_every,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tqdm": tqdm_package.__version__,
            "xgboost": xgb.__version__,
            "matplotlib": matplotlib.__version__,
            "hydra": hydra.__version__,
        },
    }
    prefix_history: list[float] = []
    starting_round = 0
    resume_booster = None
    if resume_bundle is not None:
        previous_settings, prefix_history, resume_booster = resume_bundle
        comparable = (
            "experiment", "source", "source_bytes", "source_sha256", "train_months",
            "validation_months", "train_rows", "validation_rows", "feature_columns",
            "categorical_features", "category_levels", "hyperparameters",
            "validation_month_shares", "history_metric", "history_is_weighted",
            "classification_threshold",
        )
        for key in comparable:
            if previous_settings[key] != json.loads(json.dumps(settings[key])):
                raise ValueError(f"Checkpoint setting differs from this run: {key}")
        starting_round = int(previous_settings["checkpoint_iteration"])
        if starting_round > total_rounds:
            raise ValueError("Checkpoint has more trees than the configured target.")
    checkpoint_writer = CheckpointWriter(
        run_dir / "checkpoints" / experiment.name, settings, prefix_history
    )
    if starting_round == total_rounds:
        if resume_booster is None:
            raise RuntimeError("A complete run needs a saved checkpoint model.")
        resume_booster.save_model(run_dir / f"{experiment.name}_model.json")
        history_values = prefix_history
    else:
        with tqdm(
            total=total_rounds,
            initial=starting_round,
            desc=f"Training {experiment.name}",
            unit=" trees",
            position=progress_position,
            leave=True,
            mininterval=0.5,
        ) as progress:
            model = xgb.XGBClassifier(
                **{**params, "n_estimators": total_rounds - starting_round},
                callbacks=[TrainingProgress(
                    progress, experiment.name, total_rounds, starting_round,
                    plot_every, plot_events, checkpoint_every, checkpoint_writer,
                )],
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                sample_weight_eval_set=[weights] if weights is not None else None,
                xgb_model=resume_booster,
                verbose=False,
            )
        history_values = prefix_history + model.evals_result()["validation_0"]["error"]
        model.save_model(run_dir / f"{experiment.name}_model.json")
    if len(history_values) != total_rounds:
        raise RuntimeError("Validation history does not match the requested tree count.")
    history = pd.DataFrame({
        "iteration": np.arange(1, total_rounds + 1),
        "valid_error": history_values,
    })
    history.to_csv(run_dir / f"{experiment.name}_history.csv", index=False)
    (run_dir / f"{experiment.name}_hyperparameters.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment": experiment.name,
        "final_valid_error": float(history_values[-1]),
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    if min(int(cfg.data.chunk_rows), int(cfg.model.n_estimators), int(cfg.runtime.plot_every), int(cfg.checkpoint.every)) < 1:
        raise ValueError("Data chunk size, tree count, plot interval, and checkpoint interval must be positive.")
    if int(cfg.runtime.jobs_per_model) < 0:
        raise ValueError("runtime.jobs_per_model must be zero or positive.")
    if cfg.model.eval_metric != "error":
        raise ValueError("The training curve and checkpoint history require model.eval_metric=error.")
    if cfg.runtime.live_plot not in ("auto", "on", "off"):
        raise ValueError("runtime.live_plot must be auto, on, or off.")
    backend = matplotlib.get_backend().lower()
    interactive_backend = backend not in {"agg", "pdf", "ps", "svg", "template", "cairo"} and "inline" not in backend
    if cfg.runtime.live_plot == "on" and not interactive_backend:
        raise RuntimeError(f"Matplotlib backend {backend!r} cannot show a live plot.")
    show_live_plot = interactive_backend and cfg.runtime.live_plot != "off"
    data_path = Path(to_absolute_path(str(cfg.data.path)))
    output_base = Path(to_absolute_path(str(cfg.output.base_dir)))
    resume_bundle = None
    if cfg.checkpoint.resume_from is not None:
        checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint.resume_from)))
        resume_bundle = load_checkpoint(checkpoint_path)
        names = (resume_bundle[0]["experiment"],)
        print(f"Resuming {names[0]} from round {resume_bundle[0]['checkpoint_iteration']}.")
    else:
        names = tuple(str(name) for name in cfg.runtime.models)
    if not names or len(set(names)) != len(names):
        raise ValueError("runtime.models must contain distinct protocol names.")
    if any(name not in cfg.protocols for name in names):
        raise ValueError("A selected model has no matching protocol configuration.")
    experiments = tuple(experiment_from_config(name, cfg) for name in names)
    frame = load_data(data_path, int(cfg.data.chunk_rows))
    categorical = tuple(
        name for name in frame.drop(columns="TARGET").select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )
    source_hash = file_sha256(data_path)
    run_dir = create_run_directory(output_base)
    live_plot = LiveValidationPlot(names) if show_live_plot else None
    if live_plot is not None and resume_bundle is not None:
        live_plot.seed(names[0], resume_bundle[1])
    if not show_live_plot:
        print("Live plotting is unavailable or disabled; validation history will be saved.")
    plot_events = Queue() if live_plot is not None else None
    workers = len(experiments) if cfg.runtime.parallel_models else 1
    configured_jobs = int(cfg.runtime.jobs_per_model)
    if configured_jobs:
        jobs_per_model = configured_jobs
    elif resume_bundle is not None:
        jobs_per_model = int(resume_bundle[0]["hyperparameters"]["n_jobs"])
    else:
        jobs_per_model = max(1, (os.cpu_count() or 1) // workers)
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                train_experiment,
                experiment, frame, categorical, run_dir,
                model_params, jobs_per_model, position,
                int(cfg.runtime.plot_every), plot_events, data_path, source_hash,
                int(cfg.checkpoint.every), resume_bundle,
            )
            for position, experiment in enumerate(experiments, start=1)
        ]
        if live_plot is not None and plot_events is not None:
            while not all(future.done() for future in futures) or not plot_events.empty():
                try:
                    live_plot.update(plot_events.get(timeout=0.1))
                except Empty:
                    live_plot.process_events()
        reports = [future.result() for future in futures]
    for report in sorted(reports, key=lambda item: item["experiment"]):
        print(f"{report['experiment']}: final valid misclassification rate={report['final_valid_error']:.6f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
