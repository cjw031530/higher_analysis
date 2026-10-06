"""Evaluate a saved Elastic Net model under a selected chronological protocol."""

from __future__ import annotations

import html
import json
import tempfile
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
    average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from . import WORKFLOW_VERSION
from .data import create_run_directory, file_sha256, load_data, transform_to_memmap, validation_weights
from .saga import load_final_model, predict_probability


def load_settings(model_path: Path, explicit_path: Path | None) -> dict:
    if explicit_path is None:
        if not model_path.name.endswith("_model.pkl"):
            raise ValueError("Set model.settings_path for a nonstandard model filename.")
        explicit_path = model_path.with_name(model_path.name.replace("_model.pkl", "_hyperparameters.json"))
    settings = json.loads(explicit_path.read_text(encoding="utf-8"))
    if settings.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError("The saved Elastic Net workflow version is incompatible.")
    if not settings.get("schema", {}).get("feature_columns"):
        raise ValueError("The model settings have no predictor schema.")
    return settings


def metric_values(target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None) -> dict:
    result = {
        "misclassification_rate": float(np.average((probability > 0.5) != target, weights=weights)),
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "log_loss": float(log_loss(target, probability, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability, sample_weight=weights)),
    }
    if np.unique(target).size == 2:
        result["roc_auc"] = float(roc_auc_score(target, probability, sample_weight=weights))
        result["average_precision"] = float(average_precision_score(target, probability, sample_weight=weights))
    return result


def metric_tables(predictions: pd.DataFrame, weights: np.ndarray | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = predictions["TARGET"].to_numpy()
    probability = predictions["probability"].to_numpy()
    rows = [{
        "scope": "overall_unweighted", "month": "all", "rows": len(predictions),
        "target_1_rows": int(target.sum()), "target_1_rate": float(target.mean()),
        **metric_values(target, probability, None),
    }]
    if weights is not None:
        rows.append({
            "scope": "overall_weighted", "month": "all", "rows": len(predictions),
            "target_1_rows": int(target.sum()),
            "target_1_rate": float(np.average(target, weights=weights)),
            **metric_values(target, probability, weights),
        })
    monthly = []
    for month, group in predictions.groupby("LNMON", sort=True):
        labels = group["TARGET"].to_numpy()
        scores = group["probability"].to_numpy()
        monthly.append({
            "scope": "monthly_unweighted", "month": str(month), "rows": len(group),
            "target_1_rows": int(labels.sum()), "target_1_rate": float(labels.mean()),
            **metric_values(labels, scores, None),
        })
    return pd.DataFrame(rows + monthly), pd.DataFrame(monthly)


def calibration_table(target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None) -> pd.DataFrame:
    order = np.argsort(probability, kind="stable")
    bins = np.minimum(9, np.arange(len(target)) * 10 // len(target))
    actual_weights = np.ones(len(target)) if weights is None else weights[order]
    totals = np.bincount(bins, weights=actual_weights, minlength=10)
    positive = np.bincount(bins, weights=target[order] * actual_weights, minlength=10)
    scores = np.bincount(bins, weights=probability[order] * actual_weights, minlength=10)
    return pd.DataFrame({
        "decile": np.arange(1, 11),
        "rows": np.bincount(bins, minlength=10),
        "weight_sum": totals,
        "mean_probability": np.divide(scores, totals, out=np.full(10, np.nan), where=totals > 0),
        "observed_target_rate": np.divide(positive, totals, out=np.full(10, np.nan), where=totals > 0),
    })


def threshold_table(target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None) -> pd.DataFrame:
    cutoffs = np.arange(0.1, 1.0, 0.1)
    predicted = probability[None, :] > cutoffs[:, None]
    actual_weights = np.ones(len(target)) if weights is None else weights
    positive = (target == 1) * actual_weights
    negative = (target == 0) * actual_weights
    tp = predicted @ positive
    fp = predicted @ negative
    fn = positive.sum() - tp
    tn = negative.sum() - fp

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

    return pd.DataFrame({
        "threshold": cutoffs,
        "weighted_true_positive": tp,
        "weighted_false_positive": fp,
        "weighted_true_negative": tn,
        "weighted_false_negative": fn,
        "misclassification_rate": ratio(fp + fn, tp + fp + tn + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
    })


def confusion_table(target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None) -> pd.DataFrame:
    predicted = probability > 0.5
    actual_weights = np.ones(len(target)) if weights is None else weights
    rows = []
    for actual in (0, 1):
        for prediction in (0, 1):
            selected = (target == actual) & (predicted == prediction)
            rows.append({
                "actual": actual, "predicted": prediction,
                "rows": int(selected.sum()), "weight_sum": float(actual_weights[selected].sum()),
            })
    return pd.DataFrame(rows)


def save_plots(
    output_dir: Path, history: pd.DataFrame, monthly: pd.DataFrame,
    calibration: pd.DataFrame, confusion: pd.DataFrame,
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None,
    native_weighted: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["valid_error"], linewidth=1.6)
    ax.set(
        title="Validation misclassification rate during training",
        xlabel="Completed SAGA epoch", ylabel="Validation misclassification rate",
    )
    if native_weighted:
        ax.text(0.98, 0.98, "Native month weights", transform=ax.transAxes, ha="right", va="top")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, name, label in zip(
        axes, ("misclassification_rate", "roc_auc", "log_loss"),
        ("Misclassification rate", "ROC AUC", "Log loss"),
    ):
        ax.bar(monthly["month"], monthly[name])
        ax.set(title=label, xlabel="LNMON")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Unweighted validation metrics by month")
    fig.savefig(output_dir / "monthly_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    if np.unique(target).size == 2:
        fpr, tpr, _ = roc_curve(target, probability, sample_weight=weights)
        precision, recall, _ = precision_recall_curve(target, probability, sample_weight=weights)
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].plot(recall, precision)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.plot(calibration["mean_probability"], calibration["observed_target_rate"], "o-", label="Model")
    ax.set(title="Calibration by score decile", xlabel="Mean predicted probability", ylabel="Observed target rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(0, 1, 31)
    for label in (0, 1):
        selected = target == label
        if selected.any():
            ax.hist(
                probability[selected], bins=bins,
                weights=weights[selected] if weights is not None else None,
                density=True, alpha=0.5, label=f"TARGET={label}",
            )
    ax.set(title="Score distribution by target", xlabel="Predicted probability", ylabel="Within-class density")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    values = confusion["weight_sum" if weights is not None else "rows"].to_numpy().reshape(2, 2)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues")
    ax.set(title="Month-weighted confusion matrix" if weights is not None else "Confusion matrix",
           xlabel="Predicted TARGET", ylabel="Actual TARGET", xticks=[0, 1], yticks=[0, 1])
    for actual in (0, 1):
        for predicted in (0, 1):
            ax.text(predicted, actual, f"{values[actual, predicted]:.3f}" if weights is not None else f"{values[actual, predicted]:.0f}",
                    ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_report(
    output_dir: Path, settings: dict, protocol_name: str,
    summary: pd.DataFrame, comparison: pd.DataFrame,
    calibration: pd.DataFrame, thresholds: pd.DataFrame, confusion: pd.DataFrame,
    primary_error: float,
) -> None:
    figures = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{filename}" alt="{html.escape(title)}"></section>'
        for title, filename in (
            ("Training curve", "training_curve.png"),
            ("Monthly metrics", "monthly_metrics.png"),
            ("ROC and precision-recall", "roc_pr_curves.png"),
            ("Calibration", "calibration.png"),
            ("Score distribution", "score_distribution.png"),
            ("Confusion matrix", "confusion_matrix.png"),
        )
    )
    tables = "\n".join(
        f"<section><h2>{title}</h2>{table.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</section>"
        for title, table in (
            ("Overall and monthly metrics", summary),
            ("Same-model protocol comparison", comparison),
            ("Confusion matrix", confusion),
            ("Calibration deciles", calibration),
            ("Threshold diagnostics (0.5 is the fixed final cutoff)", thresholds),
        )
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Elastic Net validation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}}
section{{margin:2rem 0}}img{{max-width:100%;height:auto;border:1px solid #d8dee5}}
table{{border-collapse:collapse;display:block;overflow-x:auto}}th,td{{padding:.45rem .7rem;border-bottom:1px solid #d8dee5;text-align:right;white-space:nowrap}}</style>
</head><body><h1>{html.escape(settings['experiment'])} Elastic Net model on {html.escape(protocol_name)} validation</h1>
<p><strong>Final validation misclassification rate at probability &gt; 0.5: {primary_error:.6f}</strong></p>
<p>The training curve uses the model's original validation protocol. Scores describe undersampled data, not an independent test set.</p>
{tables}{figures}</body></html>
"""
    (output_dir / "report.html").write_text(content, encoding="utf-8")


@hydra.main(version_base="1.3", config_path="conf", config_name="valid")
def main(cfg: DictConfig) -> None:
    if cfg.model.path is None:
        raise ValueError("Set model.path to a saved *_model.pkl file.")
    model_path = Path(to_absolute_path(str(cfg.model.path)))
    settings_path = Path(to_absolute_path(str(cfg.model.settings_path))) if cfg.model.settings_path else None
    settings = load_settings(model_path, settings_path)
    if cfg.validation.protocol == "native":
        protocol_name = settings["experiment"]
        valid_months = tuple(int(month) for month in settings["validation_months"])
        shares = settings["validation_month_shares"]
    else:
        protocol_name = str(cfg.validation.protocol)
        if protocol_name not in cfg.protocols:
            raise ValueError("validation.protocol must be native or a configured protocol.")
        spec = cfg.protocols[protocol_name]
        valid_months = tuple(int(month) for month in spec.validation_months)
        shares = spec.validation_month_shares
    shares = None if shares is None else {int(month): float(value) for month, value in shares.items()}
    overlap = sorted(set(valid_months) & set(settings["train_months"]))
    if overlap:
        raise ValueError(f"Validation months overlap the model's training months: {overlap}")
    data_path = Path(to_absolute_path(str(cfg.data.path))) if cfg.data.path else Path(settings["source"])
    if cfg.data.path is None and (
        data_path.stat().st_size != settings["source_bytes"]
        or file_sha256(data_path) != settings["source_sha256"]
    ):
        raise ValueError("The original data source changed since training.")
    frame = load_data(data_path, int(cfg.data.chunk_rows), require_all_months=False)
    indices = np.flatnonzero(frame["LNMON"].isin(valid_months).to_numpy())
    selected_months = frame["LNMON"].iloc[indices].to_numpy()
    if set(selected_months) != set(valid_months):
        raise ValueError("The validation data lacks one or more required months.")
    model = load_final_model(model_path)
    if model["schema"] != settings["schema"] or int(model["epochs"]) != settings["actual_epochs"]:
        raise ValueError("Saved model and hyperparameters disagree.")
    target = frame["TARGET"].iloc[indices].to_numpy(dtype="int8")
    weights = validation_weights(selected_months, shares)
    with tempfile.TemporaryDirectory(prefix=".elasticnet_valid_") as workspace:
        features, unseen_counts = transform_to_memmap(
            frame, indices, model["schema"], Path(workspace) / "valid.npy", show_progress=True
        )
        coef = np.asarray(model["coef"], dtype="float32")[:, None]
        intercept = np.array([model["intercept"]], dtype="float32")
        probability = predict_probability(features, coef, intercept)
    output_base = Path(to_absolute_path(str(cfg.output.base_dir))) if cfg.output.base_dir else model_path.parent / "validation"
    output_dir = create_run_directory(output_base, prefix=f"{settings['experiment']}_on_{protocol_name}")
    predictions = pd.DataFrame({
        "source_row": indices, "LNMON": selected_months, "TARGET": target,
        "probability": probability, "predicted_target": (probability > 0.5).astype("int8"),
        "validation_weight": weights if weights is not None else np.ones(len(target)),
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary, monthly = metric_tables(predictions, weights)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly_metrics.csv", index=False)
    calibration = calibration_table(target, probability, weights)
    calibration.to_csv(output_dir / "calibration_deciles.csv", index=False)
    thresholds = threshold_table(target, probability, weights)
    thresholds.to_csv(output_dir / "threshold_diagnostics.csv", index=False)
    confusion = confusion_table(target, probability, weights)
    confusion.to_csv(output_dir / "confusion_matrix.csv", index=False)
    history_path = (
        Path(to_absolute_path(str(cfg.model.history_path))) if cfg.model.history_path
        else model_path.with_name(model_path.name.replace("_model.pkl", "_history.csv"))
    )
    history = pd.read_csv(history_path)
    if list(history.columns) != ["epoch", "valid_error", "relative_coef_change"] or int(history["epoch"].iloc[-1]) != model["epochs"]:
        raise ValueError("Saved training history is incomplete or incompatible.")
    scope = "overall_weighted" if weights is not None else "overall_unweighted"
    primary_error = float(summary.loc[summary["scope"].eq(scope), "misclassification_rate"].iloc[0])
    comparison = pd.DataFrame([
        {
            "evaluation": "training_protocol", "protocol": settings["experiment"],
            "months": ",".join(str(month) for month in settings["validation_months"]),
            "month_weighted": settings["validation_month_shares"] is not None,
            "misclassification_rate": float(history["valid_error"].iloc[-1]),
        },
        {
            "evaluation": "selected_protocol", "protocol": protocol_name,
            "months": ",".join(str(month) for month in valid_months),
            "month_weighted": weights is not None,
            "misclassification_rate": primary_error,
        },
    ])
    comparison.to_csv(output_dir / "protocol_comparison.csv", index=False)
    save_plots(
        output_dir, history, monthly, calibration, confusion,
        target, probability, weights, settings["validation_month_shares"] is not None,
    )
    save_report(output_dir, settings, protocol_name, summary, comparison, calibration, thresholds, confusion, primary_error)
    metadata = {
        "model": str(model_path), "data": str(data_path.resolve()),
        "model_experiment": settings["experiment"], "validation_protocol": protocol_name,
        "validation_months": valid_months, "validation_month_shares": shares,
        "classification_threshold": 0.5, "primary_metric": "misclassification_rate",
        "primary_score": primary_error,
        "unseen_validation_categories_mapped_to_missing": unseen_counts,
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Final validation misclassification rate ({scope}, probability > 0.5): {primary_error:.6f}")
    print(summary.to_string(index=False))
    print(f"Validation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
