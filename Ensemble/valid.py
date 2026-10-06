"""Validate a saved ensemble on untouched outer months and write a report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, precision_recall_curve, roc_auc_score, roc_curve,
)
from tqdm.auto import tqdm

from Ensemble import WORKFLOW_VERSION
from Ensemble.data import (
    create_run_directory, file_sha256, load_data,
    protocol_from_config, transform_features, validation_weights,
)
from Ensemble.model import (
    FeatureSet, average_probabilities, classification_error,
    load_models, predict_members,
)


def load_settings(model_path: Path, explicit: Path | None = None) -> dict:
    if not model_path.is_dir() or not model_path.name.endswith("_model"):
        raise ValueError("model.path must be a saved v1_model or v2_model directory.")
    inferred = model_path.with_name(model_path.name.replace("_model", "_hyperparameters.json"))
    settings = json.loads((explicit or inferred).read_text(encoding="utf-8"))
    if settings.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError("The model workflow version is incompatible.")
    if model_path.name != f"{settings['experiment']}_model":
        raise ValueError("Model directory and saved experiment disagree.")
    return settings


def metric_values(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, threshold: float,
) -> dict:
    result = {
        "misclassification_rate": classification_error(target, probability, weights, threshold),
        "always_zero_error": float(np.average(target, weights=weights)),
        "log_loss": float(log_loss(target, probability, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability, sample_weight=weights)),
    }
    if len(np.unique(target)) == 2:
        result["roc_auc"] = float(roc_auc_score(target, probability, sample_weight=weights))
        result["average_precision"] = float(average_precision_score(target, probability, sample_weight=weights))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result


def summary_tables(
    months: np.ndarray, target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"scope": "overall_unweighted", "rows": len(target),
             **metric_values(target, probability, None, threshold)}]
    if weights is not None:
        rows.append({"scope": "overall_weighted", "rows": len(target),
                     **metric_values(target, probability, weights, threshold)})
    common = np.isin(months, [202404, 202405])
    if common.any() and not common.all():
        rows.append({"scope": "common_v1_unweighted", "rows": int(common.sum()),
                     **metric_values(target[common], probability[common], None, threshold)})
    monthly = []
    for month in sorted(np.unique(months)):
        selected = months == month
        monthly.append({"month": int(month), "rows": int(selected.sum()),
                        **metric_values(target[selected], probability[selected], None, threshold)})
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def calibration_table(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    index = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    mass = np.ones(len(target)) if weights is None else weights
    counts = np.bincount(index, minlength=bins)
    weighted = np.bincount(index, weights=mass, minlength=bins)
    score_sum = np.bincount(index, weights=mass * probability, minlength=bins)
    target_sum = np.bincount(index, weights=mass * target, minlength=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:], "rows": counts,
        "mean_probability": np.divide(score_sum, weighted, out=np.full(bins, np.nan), where=weighted > 0),
        "observed_positive_rate": np.divide(target_sum, weighted, out=np.full(bins, np.nan), where=weighted > 0),
    })


def threshold_table(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, selected_threshold: float,
) -> pd.DataFrame:
    cutoffs = np.unique(np.r_[np.arange(0.1, 1.0, 0.1), selected_threshold])
    wrong = (probability[None, :] > cutoffs[:, None]) != target[None, :]
    if weights is None:
        errors = wrong.mean(axis=1)
    else:
        errors = wrong @ weights / weights.sum()
    return pd.DataFrame({
        "threshold": cutoffs,
        "misclassification_rate": errors,
        "selected_for_primary_score": np.isclose(cutoffs, selected_threshold),
    })


def confusion_table(
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, threshold: float,
) -> pd.DataFrame:
    predicted = (probability > threshold).astype("int8")
    counts = confusion_matrix(target, predicted, labels=[0, 1])
    weighted = confusion_matrix(target, predicted, labels=[0, 1], sample_weight=weights)
    return pd.DataFrame([
        {"actual": actual, "predicted": guess, "rows": int(counts[actual, guess]),
         "weighted_mass": float(weighted[actual, guess])}
        for actual in (0, 1) for guess in (0, 1)
    ])


def save_plots(
    output: Path, history: pd.DataFrame, monthly: pd.DataFrame,
    calibration: pd.DataFrame, confusion: pd.DataFrame,
    target: np.ndarray, probability: np.ndarray,
    weights: np.ndarray | None, settings: dict, protocol_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label in (
        ("valid_error", "Equal ensemble"), ("catboost_error", "CatBoost"),
        ("lightgbm_error", "LightGBM"), ("xgboost_error", "XGBoost"),
    ):
        ax.plot(history["round"], history[column], label=label)
    ax.set(xlabel="Boosting rounds", ylabel="Misclassification rate",
           title=f"Training-time outer validation ({settings['experiment']} protocol)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(monthly["month"].astype(str), monthly["misclassification_rate"])
    axes[0].set(title="Monthly misclassification", ylabel="0/1 loss")
    axes[1].bar(monthly["month"].astype(str), monthly["always_zero_error"])
    axes[1].set(title="Monthly always-zero baseline", ylabel="0/1 loss")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "monthly_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(np.unique(target)) == 2:
        false_positive, true_positive, _ = roc_curve(target, probability, sample_weight=weights)
        precision, recall, _ = precision_recall_curve(target, probability, sample_weight=weights)
        axes[0].plot(false_positive, true_positive)
        axes[1].plot(recall, precision)
    axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(output / "roc_pr.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    nonempty = calibration["rows"] > 0
    ax.plot(calibration.loc[nonempty, "mean_probability"],
            calibration.loc[nonempty, "observed_positive_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate",
           title="Calibration on undersampled data")
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in (0, 1):
        values = probability[target == label]
        if len(values):
            ax.hist(values, bins=25, density=True, alpha=0.5, label=f"TARGET={label}")
    ax.set(xlabel="Ensemble probability", ylabel="Density", title="Score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "score_distribution.png", dpi=150)
    plt.close(fig)

    matrix = confusion.pivot(index="actual", columns="predicted", values="rows").to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    for actual in (0, 1):
        for predicted in (0, 1):
            ax.text(predicted, actual, f"{matrix[actual, predicted]:,}",
                    ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix ({protocol_name})")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_report(output: Path, model_path: Path, settings: dict, protocol_name: str,
                summary: pd.DataFrame, monthly: pd.DataFrame,
                comparison: pd.DataFrame, confusion: pd.DataFrame,
                primary_scope: str, primary_error: float) -> None:
    tables = [
        ("Overall metrics", summary), ("Monthly metrics", monthly),
        ("Ensemble and member comparison", comparison), ("Confusion matrix", confusion),
    ]
    sections = "\n".join(
        f"<section><h2>{html.escape(title)}</h2>{table.to_html(index=False, float_format=lambda value: f'{value:.6f}')}</section>"
        for title, table in tables
    )
    images = "\n".join(
        f'<section><img src="{name}.png" alt="{name.replace("_", " ")}"></section>'
        for name in ("training_curve", "monthly_metrics", "roc_pr", "calibration",
                     "score_distribution", "confusion_matrix")
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Ensemble validation</title><style>
body {{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table {{border-collapse:collapse;display:block;overflow:auto}} td,th {{padding:.35rem .6rem;border:1px solid #ddd}}
img {{max-width:100%;height:auto}} section {{margin:2rem 0}}
</style></head><body><h1>Ensemble validation</h1>
<p>Model: {html.escape(str(model_path.resolve()))}<br>Training protocol: {settings['experiment']}<br>
Evaluation protocol: {html.escape(protocol_name)}. No independent test set is available.</p>
<p><strong>Final validation 0/1 loss / misclassification rate ({primary_scope}): {primary_error:.6f}</strong></p>
<p>The fixed classification rule is probability &gt; 0.5. Outer validation scores were displayed during training but did not choose hyperparameters, rounds, or checkpoints. Probabilities are scores on an undersampled source dataset.</p>
{sections}{images}</body></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")


def run_validation(cfg: DictConfig) -> Path:
    if not cfg.model.path:
        raise ValueError("model.path is required.")
    model_path = Path(to_absolute_path(str(cfg.model.path)))
    settings_path = Path(to_absolute_path(str(cfg.model.settings_path))) if cfg.model.settings_path else None
    settings = load_settings(model_path, settings_path)
    data_path = Path(to_absolute_path(str(cfg.data.path))) if cfg.data.path else Path(settings["source"])
    if (data_path.stat().st_size != settings["source_bytes"]
            or file_sha256(data_path) != settings["source_sha256"]):
        raise ValueError("The source CSV does not match the file used for training.")
    protocol_name = str(cfg.validation.protocol)
    if protocol_name == "native":
        protocol_name = settings["experiment"]
        valid_months = tuple(settings["validation_months"])
        shares = settings["validation_month_shares"]
        shares = None if shares is None else {int(key): float(value) for key, value in shares.items()}
    elif protocol_name in ("v1", "v2"):
        protocol = protocol_from_config(protocol_name, cfg)
        valid_months = protocol.validation_months
        shares = protocol.validation_month_shares
    else:
        raise ValueError("validation.protocol must be native, v1, or v2.")
    if set(valid_months) & set(settings["train_months"]):
        raise ValueError("Evaluation months overlap this model's training months.")
    frame = load_data(data_path, int(cfg.data.chunk_rows), months_filter=valid_months)
    valid_frame = frame.loc[frame["LNMON"].isin(valid_months)]
    if valid_frame.empty or set(valid_frame["LNMON"].unique()) != set(valid_months):
        raise ValueError("Required evaluation months are absent.")
    threads = int(cfg.runtime.threads_per_member) or max(1, (os.cpu_count() or 1) // 3)
    if threads < 1:
        raise ValueError("runtime.threads_per_member cannot be negative.")
    target = valid_frame["TARGET"].to_numpy(dtype="int8")
    months = valid_frame["LNMON"].to_numpy(dtype="int32")
    weights = validation_weights(months, shares)
    threshold = float(settings["hyperparameters"]["decision_threshold"])
    output_base = (Path(to_absolute_path(str(cfg.output.base_dir))) if cfg.output.base_dir
                   else model_path.parent / "validation")
    output = create_run_directory(output_base, prefix=f"{settings['experiment']}_on_{protocol_name}")
    with tqdm(total=4, desc="Validating saved ensemble", unit=" steps") as progress:
        native, encoded = transform_features(valid_frame, settings["schema"])
        features = FeatureSet(native, encoded, target, settings["schema"]["categorical_features"])
        progress.update(1)
        models = load_models(model_path, threads)
        probabilities = predict_members(models, features, threads)
        ensemble = average_probabilities(probabilities)
        progress.update(1)
        summary, monthly = summary_tables(months, target, ensemble, weights, threshold)
        primary_scope = "overall_weighted" if weights is not None else "overall_unweighted"
        primary_error = float(summary.loc[summary["scope"].eq(primary_scope), "misclassification_rate"].iloc[0])
        comparison = []
        for name, score in (("ensemble", ensemble), *probabilities.items()):
            comparison.append({"model": name, "rows": len(target),
                               **metric_values(target, score, weights, threshold)})
        comparison.append({"model": "always_zero", "rows": len(target),
                           **metric_values(target, np.zeros(len(target)), weights, threshold)})
        if cfg.comparison_model.path:
            other_path = Path(to_absolute_path(str(cfg.comparison_model.path)))
            other_settings = load_settings(other_path)
            if (other_settings["source_sha256"] != settings["source_sha256"]
                    or set(valid_months) & set(other_settings["train_months"])):
                raise ValueError("Comparison model source or training months are incompatible.")
            other_native, other_encoded = transform_features(valid_frame, other_settings["schema"])
            other_features = FeatureSet(other_native, other_encoded, target,
                                        other_settings["schema"]["categorical_features"])
            other_scores = predict_members(load_models(other_path, threads), other_features, threads)
            comparison.append({
                "model": f"{other_settings['experiment']}_comparison_ensemble", "rows": len(target),
                **metric_values(target, average_probabilities(other_scores), weights, threshold),
            })
        comparison_frame = pd.DataFrame(comparison)
        calibration = calibration_table(target, ensemble, weights)
        thresholds = threshold_table(target, ensemble, weights, threshold)
        confusion = confusion_table(target, ensemble, weights, threshold)
        summary.to_csv(output / "metrics_summary.csv", index=False)
        monthly.to_csv(output / "monthly_metrics.csv", index=False)
        comparison_frame.to_csv(output / "model_comparison.csv", index=False)
        calibration.to_csv(output / "calibration.csv", index=False)
        thresholds.to_csv(output / "threshold_diagnostics.csv", index=False)
        confusion.to_csv(output / "confusion_matrix.csv", index=False)
        progress.update(1)
        history_path = (Path(to_absolute_path(str(cfg.model.history_path))) if cfg.model.history_path
                        else model_path.with_name(model_path.name.replace("_model", "_history.csv")))
        history = pd.read_csv(history_path)
        if (list(history.columns) != ["round", "valid_error", "catboost_error", "lightgbm_error", "xgboost_error"]
                or int(history["round"].iloc[-1]) != int(settings["selected_rounds"])):
            raise ValueError("The saved training curve is incomplete or incompatible.")
        if cfg.validation.protocol == "native" and not np.isclose(
            float(history["valid_error"].iloc[-1]), primary_error, rtol=0, atol=1e-9,
        ):
            raise ValueError("Saved model predictions disagree with the final native training score.")
        save_plots(output, history, monthly, calibration, confusion, target, ensemble,
                   weights, settings, protocol_name)
        save_report(output, model_path, settings, protocol_name, summary, monthly,
                    comparison_frame, confusion, primary_scope, primary_error)
        (output / "validation_metadata.json").write_text(json.dumps({
            "model": str(model_path.resolve()), "source": str(data_path.resolve()),
            "model_experiment": settings["experiment"], "validation_protocol": protocol_name,
            "validation_months": valid_months, "validation_month_shares": shares,
            "classification_threshold": threshold,
            "primary_metric": "misclassification_rate", "primary_scope": primary_scope,
            "primary_score": primary_error, "primary_loss": primary_error,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        progress.update(1)
    print(f"Final validation 0/1 loss / misclassification rate ({primary_scope}): {primary_error:.6f}")
    print(summary.to_string(index=False))
    print(f"Validation report saved to: {output}")
    return output


@hydra.main(version_base="1.3", config_path="conf", config_name="valid")
def main(cfg: DictConfig) -> None:
    run_validation(cfg)


if __name__ == "__main__":
    main()
