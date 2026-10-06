"""Train two time-based XGBoost classifiers and report validation results."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tqdm as tqdm_package
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "kcb_202306_202405_undersampled_1to4.csv"
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "outputs"
ALL_MONTHS = (
    202306, 202307, 202308, 202309, 202310, 202311,
    202312, 202401, 202402, 202403, 202404, 202405,
)
V2_MONTH_SHARES = {202403: 0.6, 202404: 0.3, 202405: 0.1}
WORKFLOW_VERSION = "1.0.0"


@dataclass(frozen=True)
class Experiment:
    name: str
    train_months: tuple[int, ...]
    valid_months: tuple[int, ...]
    weighted_validation: bool


EXPERIMENTS = (
    Experiment("v1", ALL_MONTHS[:10], ALL_MONTHS[10:], False),
    Experiment("v2", ALL_MONTHS[:9], ALL_MONTHS[9:], True),
)


class TrainingProgress(xgb.callback.TrainingCallback):
    """Update one tqdm bar after each compiled boosting iteration."""

    def __init__(self, progress: tqdm) -> None:
        self.progress = progress

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict
    ) -> bool:
        self.progress.update(epoch + 1 - self.progress.n)
        return False


def load_data(path: Path, chunk_rows: int) -> pd.DataFrame:
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
    if observed != set(ALL_MONTHS):
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


def validation_weights(months: pd.Series, weighted: bool) -> np.ndarray | None:
    """Give each requested month its exact total share in validation."""

    if not weighted:
        return None
    counts = months.value_counts()
    if set(counts.index) != set(V2_MONTH_SHARES):
        raise ValueError("Weighted validation requires all three validation months.")
    weight_by_month = {
        month: share / int(counts[month])
        for month, share in V2_MONTH_SHARES.items()
    }
    return months.map(weight_by_month).to_numpy(dtype="float64")


def calculate_metrics(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float | None]:
    """Calculate ranking and probability metrics, including weighted variants."""

    if len(target) == 0:
        return {"roc_auc": None, "average_precision": None, "log_loss": None}
    result: dict[str, float | None] = {
        "roc_auc": None,
        "average_precision": float(
            average_precision_score(target, probability, sample_weight=weights)
        ) if len(np.unique(target)) == 2 else None,
        "log_loss": float(log_loss(target, probability, sample_weight=weights, labels=[0, 1])),
    }
    if len(np.unique(target)) == 2:
        result["roc_auc"] = float(roc_auc_score(target, probability, sample_weight=weights))
    return result


def create_run_directory(base: Path) -> Path:
    """Create a new output directory without replacing an earlier experiment."""

    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        run_dir = base / (f"run_{stamp}" if suffix == 0 else f"run_{stamp}_{suffix}")
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
    n_estimators: int,
    jobs_per_model: int,
    progress_position: int,
) -> dict:
    """Fit one model and write its model, predictions, and metrics."""

    train = frame.loc[frame["LNMON"].isin(experiment.train_months)]
    valid = frame.loc[frame["LNMON"].isin(experiment.valid_months)]
    x_train, x_valid, unseen_counts = prepare_features(train, valid, categorical)
    y_train = train["TARGET"].to_numpy()
    y_valid = valid["TARGET"].to_numpy()
    weights = validation_weights(valid["LNMON"], experiment.weighted_validation)

    params = {
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.04,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": jobs_per_model,
    }
    with tqdm(
        total=n_estimators,
        desc=f"Training {experiment.name}",
        unit=" trees",
        position=progress_position,
        leave=True,
        mininterval=0.5,
    ) as progress:
        model = xgb.XGBClassifier(**params, callbacks=[TrainingProgress(progress)])
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            sample_weight_eval_set=[weights] if weights is not None else None,
            verbose=False,
        )

    probability = model.predict_proba(x_valid)[:, 1]
    predictions = pd.DataFrame({
        "source_row": valid.index.to_numpy(),
        "LNMON": valid["LNMON"].to_numpy(),
        "TARGET": y_valid,
        "probability": probability,
        "validation_weight": weights if weights is not None else np.ones(len(valid)),
    })
    predictions.to_csv(run_dir / f"{experiment.name}_validation_predictions.csv", index=False)
    model.save_model(run_dir / f"{experiment.name}_model.json")

    by_month = {}
    for month, group in predictions.groupby("LNMON", sort=True):
        by_month[str(month)] = {
            "rows": len(group),
            "target_1_rows": int(group["TARGET"].sum()),
            "metrics": calculate_metrics(
                group["TARGET"].to_numpy(), group["probability"].to_numpy()
            ),
        }
    report = {
        "experiment": experiment.name,
        "workflow_version": WORKFLOW_VERSION,
        "train_months": experiment.train_months,
        "validation_months": experiment.valid_months,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "feature_count": x_train.shape[1],
        "categorical_features": categorical,
        "unseen_validation_categories_mapped_to_missing": unseen_counts,
        "parameters": params,
        "overall_unweighted": calculate_metrics(y_valid, probability),
        "overall_weighted": calculate_metrics(y_valid, probability, weights)
        if weights is not None else None,
        "validation_month_shares": V2_MONTH_SHARES if weights is not None else None,
        "by_month": by_month,
    }
    (run_dir / f"{experiment.name}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--chunk-rows", type=int, default=20_000)
    parser.add_argument("--n-estimators", type=int, default=1600)
    parser.add_argument(
        "--jobs-per-model", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="XGBoost CPU threads per model; two models run in parallel by default.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Train models one after the other when memory is limited.",
    )
    args = parser.parse_args()
    if args.chunk_rows < 1 or args.n_estimators < 1 or args.jobs_per_model < 1:
        parser.error("--chunk-rows, --n-estimators, and --jobs-per-model must be positive.")
    return args


def main() -> None:
    args = parse_args()
    frame = load_data(args.data, args.chunk_rows)
    categorical = tuple(
        name for name in frame.drop(columns="TARGET").select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )
    run_dir = create_run_directory(args.output_base)
    run_info = {
        "workflow_version": WORKFLOW_VERSION,
        "source": str(args.data.resolve()),
        "source_bytes": args.data.stat().st_size,
        "rows": len(frame),
        "feature_columns": [name for name in frame if name != "TARGET"],
        "categorical_features": categorical,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tqdm": tqdm_package.__version__,
            "xgboost": xgb.__version__,
        },
        "parallel_models": not args.sequential,
        "jobs_per_model": args.jobs_per_model,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_info, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if args.sequential:
        reports = [
            train_experiment(
                experiment, frame, categorical, run_dir,
                args.n_estimators, args.jobs_per_model, position,
            )
            for position, experiment in enumerate(EXPERIMENTS, start=1)
        ]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    train_experiment,
                    experiment, frame, categorical, run_dir,
                    args.n_estimators, args.jobs_per_model, position,
                ): experiment.name
                for position, experiment in enumerate(EXPERIMENTS, start=1)
            }
            reports = [future.result() for future in as_completed(futures)]
    for report in sorted(reports, key=lambda item: item["experiment"]):
        metric_name = "overall_weighted" if report["overall_weighted"] else "overall_unweighted"
        print(f"{report['experiment']}: {metric_name}={report[metric_name]}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
