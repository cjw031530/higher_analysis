"""Validate a saved XGBoost model and create diagnostic tables and figures."""

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
import xgboost as xgb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .train import create_run_directory, file_sha256, load_data


def load_settings(model_path: Path, settings_path: Path | None) -> dict:
    if settings_path is None:
        if not model_path.name.endswith("_model.json"):
            raise ValueError("Use --settings when the model file name is not '*_model.json'.")
        settings_path = model_path.with_name(
            model_path.name.replace("_model.json", "_hyperparameters.json")
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings.get("workflow_version") not in {"2.0.0", "3.0.0", "3.1.0"}:
        raise ValueError("This validator requires a version 2.0.0, 3.0.0, or 3.1.0 training artifact.")
    if not settings.get("feature_columns") or not settings.get("validation_months"):
        raise ValueError("The settings file lacks its feature schema or validation months.")
    return settings


def validation_frame(
    frame: pd.DataFrame, settings: dict, validation_months: tuple[int, ...]
) -> tuple[pd.DataFrame, dict[str, int]]:
    expected_columns = settings["feature_columns"]
    actual_columns = [name for name in frame if name != "TARGET"]
    if actual_columns != expected_columns:
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(f"Predictor schema differs from training: missing={missing}, extra={extra}.")
    months = set(validation_months)
    selected = frame.loc[frame["LNMON"].isin(months)]
    if set(selected["LNMON"].unique()) != months:
        raise ValueError("The validation CSV is missing one or more required months.")
    features = selected[expected_columns].copy(deep=False)
    unseen_counts = {}
    for name, levels in settings["category_levels"].items():
        values = features[name].astype("string")
        unseen = values.notna() & ~values.isin(levels)
        unseen_counts[name] = int(unseen.sum())
        features[name] = values.mask(unseen).astype(pd.CategoricalDtype(categories=levels))
    return features, unseen_counts


def month_weights(months: pd.Series, shares: dict | None) -> np.ndarray | None:
    if shares is None:
        return None
    shares = {int(month): float(share) for month, share in shares.items()}
    counts = months.value_counts()
    if set(counts.index) != set(shares) or not np.isclose(sum(shares.values()), 1.0):
        raise ValueError("Validation month shares do not match the available months.")
    per_row = {month: share / int(counts[month]) for month, share in shares.items()}
    return months.map(per_row).to_numpy(dtype="float64")


def metrics(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "misclassification_rate": float(np.average((probability > 0.5) != target, weights=weights)),
        "roc_auc": None,
        "average_precision": None,
        "log_loss": float(log_loss(target, probability, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability, sample_weight=weights)),
    }
    if len(np.unique(target)) == 2:
        result["roc_auc"] = float(roc_auc_score(target, probability, sample_weight=weights))
        result["average_precision"] = float(
            average_precision_score(target, probability, sample_weight=weights)
        )
    return result


def metric_tables(
    predictions: pd.DataFrame, weights: np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cases = [("overall_unweighted", predictions, None)]
    if weights is not None:
        cases.append(("overall_weighted", predictions, weights))
    for name, group, case_weights in cases:
        labels = group["TARGET"].to_numpy()
        values = group["probability"].to_numpy()
        rows.append({
            "scope": name,
            "month": "all",
            "rows": len(group),
            "target_1_rows": int(labels.sum()),
            "target_1_rate": float(np.average(labels, weights=case_weights)),
            **metrics(labels, values, case_weights),
        })
    monthly_rows = []
    for month, group in predictions.groupby("LNMON", sort=True):
        labels = group["TARGET"].to_numpy()
        values = group["probability"].to_numpy()
        monthly_rows.append({
            "scope": "monthly_unweighted",
            "month": str(month),
            "rows": len(group),
            "target_1_rows": int(labels.sum()),
            "target_1_rate": float(labels.mean()),
            **metrics(labels, values, None),
        })
    return pd.DataFrame(rows + monthly_rows), pd.DataFrame(monthly_rows)


def calibration_table(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None
) -> pd.DataFrame:
    """Summarize weighted observed and predicted rates in ten equal-row bins."""

    order = np.argsort(probability, kind="stable")
    count = len(target)
    bin_by_rank = np.minimum(9, np.arange(count) * 10 // count)
    actual_weights = np.ones(count) if weights is None else weights[order]
    weight_total = np.bincount(bin_by_rank, weights=actual_weights, minlength=10)
    label_total = np.bincount(bin_by_rank, weights=target[order] * actual_weights, minlength=10)
    score_total = np.bincount(bin_by_rank, weights=probability[order] * actual_weights, minlength=10)
    row_total = np.bincount(bin_by_rank, minlength=10)
    return pd.DataFrame({
        "decile": np.arange(1, 11),
        "rows": row_total,
        "weight_sum": weight_total,
        "mean_probability": np.divide(score_total, weight_total, out=np.full(10, np.nan), where=weight_total > 0),
        "observed_target_rate": np.divide(label_total, weight_total, out=np.full(10, np.nan), where=weight_total > 0),
    })


def threshold_table(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None
) -> pd.DataFrame:
    """Compute diagnostic threshold measures without selecting a cutoff."""

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
        return np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
        )

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
        "predicted_positive_share": ratio(tp + fp, tp + fp + tn + fn),
    })


def save_training_curve(history_path: Path, destination: Path, weighted: bool, metric: str) -> None:
    history = pd.read_csv(history_path)
    column = {"error": "valid_error", "logloss": "valid_logloss"}.get(metric)
    if column is None or list(history.columns) != ["iteration", column]:
        raise ValueError("Unexpected training history columns.")
    label = "Validation misclassification rate" if metric == "error" else "Validation log loss"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["iteration"], history[column], linewidth=1.6)
    ax.set(xlabel="Boosting iteration", ylabel=label, title=f"{label} during training")
    if weighted:
        ax.text(0.98, 0.98, "Month weighted", transform=ax.transAxes, ha="right", va="top")
    ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_roc_pr(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None, destination: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    if len(np.unique(target)) == 2:
        false_positive, true_positive, _ = roc_curve(target, probability, sample_weight=weights)
        precision, recall, _ = precision_recall_curve(target, probability, sample_weight=weights)
        axes[0].plot(false_positive, true_positive, label="Model")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        axes[1].plot(recall, precision, label="Model")
        prevalence = float(np.average(target, weights=weights))
        axes[1].axhline(prevalence, linestyle="--", color="gray", label="Prevalence")
        axes[0].legend()
        axes[1].legend()
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Both target classes are required", ha="center", va="center")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_calibration(calibration: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.plot(calibration["mean_probability"], calibration["observed_target_rate"], marker="o", label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed target rate", title="Calibration by score decile", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_monthly_metrics(monthly: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ("misclassification_rate", "roc_auc", "log_loss"),
        ("Misclassification rate", "ROC AUC", "Log loss"),
    ):
        ax.bar(monthly["month"], monthly[column])
        ax.set(title=title, xlabel="LNMON")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Unweighted validation metrics by month")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_score_distribution(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray | None, destination: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(0, 1, 31)
    for label in (0, 1):
        selected = target == label
        if selected.any():
            ax.hist(
                probability[selected], bins=bins, weights=weights[selected] if weights is not None else None,
                density=True, alpha=0.5, label=f"TARGET={label}",
            )
    ax.set(xlabel="Predicted probability", ylabel="Within-class density", title="Score distribution by target")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_html_report(
    destination: Path,
    settings: dict,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    """Place the validation tables and figures in one readable local report."""

    sections = [
        ("Training history", "training_curve.png"),
        ("ROC and precision-recall", "roc_pr_curves.png"),
        ("Calibration", "calibration.png"),
        ("Monthly metrics", "monthly_metrics.png"),
        ("Score distribution", "score_distribution.png"),
    ]
    figures = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{file_name}" alt="{html.escape(title)}"></section>'
        for title, file_name in sections
    )
    primary_scope = "overall_weighted" if settings["validation_month_shares"] is not None else "overall_unweighted"
    primary_error = float(summary.loc[summary["scope"].eq(primary_scope), "misclassification_rate"].iloc[0])
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(settings['experiment'])} model on {html.escape(settings['validation_protocol'])} validation</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #17202a; }}
section {{ margin: 2rem 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #d8dee5; }}
table {{ border-collapse: collapse; width: 100%; display: block; overflow-x: auto; }}
th, td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid #d8dee5; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>{html.escape(settings['experiment'])} model on {html.escape(settings['validation_protocol'])} validation</h1>
<p><strong>Final validation misclassification rate ({primary_scope}, probability &gt; 0.5): {primary_error:.6f}</strong></p>
<p>Validation months: {', '.join(str(month) for month in settings['validation_months'])}.
The training-history figure reflects the model's original training validation protocol.
The results describe the supplied undersampled data and are not independent test scores.</p>
<section><h2>Overall and monthly metrics</h2>{summary.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</section>
<section><h2>Same-model protocol comparison</h2><p>The first row is the metric recorded during training;
the second row scores the selected validation protocol with the same metric.</p>
{comparison.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</section>
{figures}
<section><h2>Calibration deciles</h2>{calibration.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</section>
<section><h2>Threshold diagnostics</h2><p>The final classification score uses 0.5; other thresholds are diagnostics only.</p>
{thresholds.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</section>
</body>
</html>
"""
    destination.write_text(report, encoding="utf-8")


@hydra.main(version_base="1.3", config_path="conf", config_name="valid")
def main(cfg: DictConfig) -> None:
    if cfg.model.path is None:
        raise ValueError("Set model.path to a saved *_model.json file.")
    if int(cfg.data.chunk_rows) < 1 or int(cfg.runtime.n_jobs) < 0:
        raise ValueError("data.chunk_rows must be positive and runtime.n_jobs cannot be negative.")
    model_path = Path(to_absolute_path(str(cfg.model.path)))
    settings_path = (
        Path(to_absolute_path(str(cfg.model.settings_path)))
        if cfg.model.settings_path is not None else None
    )
    settings = load_settings(model_path, settings_path)
    if cfg.validation.protocol == "native":
        protocol_name = settings["experiment"]
        validation_months = tuple(int(month) for month in settings["validation_months"])
        saved_shares = settings.get("validation_month_shares")
        month_shares = (
            None if saved_shares is None
            else {int(month): float(share) for month, share in saved_shares.items()}
        )
    else:
        protocol_name = str(cfg.validation.protocol)
        if protocol_name not in cfg.protocols:
            raise ValueError("validation.protocol must be native or a configured protocol name.")
        protocol = cfg.protocols[protocol_name]
        validation_months = tuple(int(month) for month in protocol.validation_months)
        month_shares = (
            None if protocol.validation_month_shares is None
            else {int(month): float(share) for month, share in protocol.validation_month_shares.items()}
        )
    overlap = sorted(set(validation_months) & set(settings["train_months"]))
    if overlap:
        raise ValueError(f"Selected validation months were used to train this model: {overlap}")
    data_path = (
        Path(to_absolute_path(str(cfg.data.path)))
        if cfg.data.path is not None else Path(settings["source"])
    )
    if cfg.data.path is None:
        if data_path.stat().st_size != settings["source_bytes"]:
            raise ValueError("The original data file size changed since training.")
        if settings.get("source_sha256") and file_sha256(data_path) != settings["source_sha256"]:
            raise ValueError("The original data file content changed since training.")
    frame = load_data(data_path, int(cfg.data.chunk_rows), require_all_months=False)
    features, unseen_counts = validation_frame(frame, settings, validation_months)
    selected = frame.loc[frame["LNMON"].isin(validation_months)]
    target = selected["TARGET"].to_numpy()
    weights = month_weights(selected["LNMON"], month_shares)
    model = xgb.XGBClassifier(n_jobs=int(cfg.runtime.n_jobs) or os.cpu_count() or 1)
    model.load_model(model_path)
    probability = model.predict_proba(features)[:, 1]

    output_base = (
        Path(to_absolute_path(str(cfg.output.base_dir)))
        if cfg.output.base_dir is not None else model_path.parent / "validation"
    )
    output_dir = create_run_directory(
        output_base, prefix=f"{settings['experiment']}_on_{protocol_name}"
    )
    predictions = pd.DataFrame({
        "source_row": selected.index.to_numpy(),
        "LNMON": selected["LNMON"].to_numpy(),
        "TARGET": target,
        "probability": probability,
        "validation_weight": weights if weights is not None else np.ones(len(target)),
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary, monthly = metric_tables(predictions, weights)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    primary_weights = weights
    calibration = calibration_table(target, probability, primary_weights)
    calibration.to_csv(output_dir / "calibration_deciles.csv", index=False)
    thresholds = threshold_table(target, probability, primary_weights)
    thresholds.to_csv(output_dir / "threshold_diagnostics.csv", index=False)
    history_path = (
        Path(to_absolute_path(str(cfg.model.history_path)))
        if cfg.model.history_path is not None
        else model_path.with_name(model_path.name.replace("_model.json", "_history.csv"))
    )
    history = pd.read_csv(history_path)
    primary_scope = "overall_weighted" if weights is not None else "overall_unweighted"
    primary_row = summary.loc[summary["scope"].eq(primary_scope)].iloc[0]
    primary_error = float(primary_row["misclassification_rate"])
    history_metric = settings["history_metric"]
    history_column = {"error": "valid_error", "logloss": "valid_logloss"}.get(history_metric)
    selected_column = {"error": "misclassification_rate", "logloss": "log_loss"}.get(history_metric)
    if history_column is None or list(history.columns) != ["iteration", history_column]:
        raise ValueError("Unexpected training history metric or columns.")
    comparison = pd.DataFrame([
        {
            "evaluation": "training_protocol",
            "protocol": settings["experiment"],
            "months": ",".join(str(month) for month in settings["validation_months"]),
            "month_weighted": bool(settings["history_is_weighted"]),
            "metric": history_metric,
            "score": float(history[history_column].iloc[-1]),
        },
        {
            "evaluation": "selected_protocol",
            "protocol": protocol_name,
            "months": ",".join(str(month) for month in validation_months),
            "month_weighted": weights is not None,
            "metric": history_metric,
            "score": float(primary_row[selected_column]),
        },
    ])
    comparison.to_csv(output_dir / "protocol_comparison.csv", index=False)
    save_training_curve(history_path, output_dir / "training_curve.png", bool(settings["history_is_weighted"]), history_metric)
    save_roc_pr(target, probability, primary_weights, output_dir / "roc_pr_curves.png")
    save_calibration(calibration, output_dir / "calibration.png")
    save_monthly_metrics(monthly, output_dir / "monthly_metrics.png")
    save_score_distribution(target, probability, primary_weights, output_dir / "score_distribution.png")
    report_settings = {
        **settings,
        "validation_protocol": protocol_name,
        "validation_months": validation_months,
        "validation_month_shares": month_shares,
    }
    save_html_report(output_dir / "report.html", report_settings, summary, comparison, calibration, thresholds)
    metadata = {
        "model": str(model_path),
        "data": str(data_path.resolve()),
        "model_experiment": settings["experiment"],
        "validation_protocol": protocol_name,
        "validation_months": validation_months,
        "validation_month_shares": month_shares,
        "weighted_diagnostics": weights is not None,
        "classification_threshold": 0.5,
        "primary_metric": "misclassification_rate",
        "primary_score": primary_error,
        "unseen_validation_categories_mapped_to_missing": unseen_counts,
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Final validation misclassification rate ({primary_scope}, probability > 0.5): {primary_error:.6f}")
    print(summary.to_string(index=False))
    print(f"Validation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
