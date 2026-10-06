"""Small synthetic checks for leakage, checkpoints, scores, and reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.naive_bayes import GaussianNB

from GaussianNB import WORKFLOW_VERSION
from GaussianNB.data import (
    ALL_MONTHS, file_sha256, fit_schema, transform_batch, validation_weights,
)
from GaussianNB.model import (
    choose_var_smoothing, encode_to_memmap, load_checkpoint,
    predict_matrix_probabilities, predict_probabilities,
    save_checkpoint, save_final_model,
)
from GaussianNB.valid import metric_tables, run_validation


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for month_index, month in enumerate(ALL_MONTHS):
        for row_index in range(12):
            target = row_index % 2
            rows.append({
                "LNMON": month,
                "TARGET": target,
                "NUMERIC": np.nan if row_index == 0 else target * 2.0 + row_index / 20,
                "CATEGORICAL": "new" if month >= 202404 and row_index == 0 else ("yes" if target else "no"),
            })
    return pd.DataFrame(rows)


class GaussianNBPipelineTests(unittest.TestCase):
    def test_train_only_schema_and_month_weights(self) -> None:
        frame = synthetic_frame()
        train = np.flatnonzero(frame["LNMON"].to_numpy() <= 202402)
        schema = fit_schema(frame, train)
        self.assertEqual(schema["feature_columns"], ["LNMON", "NUMERIC", "CATEGORICAL"])
        self.assertNotIn("new", schema["categorical_levels"]["CATEGORICAL"])
        self.assertIn("NUMERIC", schema["missing_indicator_columns"])
        valid = frame.iloc[[0, len(frame) - 12]]
        encoded = transform_batch(valid, schema)
        self.assertTrue(np.isfinite(encoded).all())
        self.assertEqual(encoded[1, schema["encoded_columns"].index("CATEGORICAL=<MISSING_OR_UNKNOWN>")], 1)
        months = np.array([202403, 202403, 202404, 202405, 202405])
        weights = validation_weights(months, {202403: 0.6, 202404: 0.3, 202405: 0.1})
        self.assertTrue(np.allclose([weights[:2].sum(), weights[2], weights[3:].sum()], [0.6, 0.3, 0.1]))

    def test_smoothing_selection_and_checkpoint_replay(self) -> None:
        frame = synthetic_frame()
        months = frame["LNMON"].to_numpy()
        target = frame["TARGET"].to_numpy()
        selected, results = choose_var_smoothing(
            frame, months, target, tuple(ALL_MONTHS[:9]), [1e-9, 1e-6],
            holdout_months=1, batch_rows=17, threshold=0.5, label="synthetic",
        )
        self.assertIn(selected, (1e-9, 1e-6))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(0 <= row["holdout_error"] <= 1 for row in results))

        schema = fit_schema(frame, np.arange(36))
        first = transform_batch(frame.iloc[:12], schema)
        second = transform_batch(frame.iloc[12:24], schema)
        reference = GaussianNB(var_smoothing=selected)
        reference.partial_fit(first, target[:12], classes=np.array([0, 1]))
        reference.partial_fit(second, target[12:24])
        first_stage = GaussianNB(var_smoothing=selected)
        first_stage.partial_fit(first, target[:12], classes=np.array([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_checkpoint(
                root, first_stage, schema, {"workflow_version": WORKFLOW_VERSION},
                [{"update": 1, "train_rows": 12, "valid_error": 0.2}],
            )
            settings, resumed, saved_schema, history = load_checkpoint(checkpoint)
            self.assertEqual(settings["workflow_version"], WORKFLOW_VERSION)
            self.assertEqual(saved_schema, schema)
            self.assertEqual(len(history), 1)
            resumed.partial_fit(second, target[12:24])
            np.testing.assert_allclose(resumed.theta_, reference.theta_, rtol=0, atol=0)
            np.testing.assert_allclose(resumed.var_, reference.var_, rtol=0, atol=0)
            np.testing.assert_allclose(resumed.predict_proba(second), reference.predict_proba(second), rtol=0, atol=0)
            validation_indices = np.arange(24, 36)
            direct = predict_probabilities(resumed, frame, validation_indices, schema, batch_rows=5)
            matrix = encode_to_memmap(
                frame, validation_indices, schema, batch_rows=5,
                path=root / "validation.npy", label="synthetic",
            )
            cached = predict_matrix_probabilities(resumed, matrix, batch_rows=5)
            np.testing.assert_allclose(cached, direct, rtol=0, atol=0)

    def test_saved_model_validation_and_v2_on_v1(self) -> None:
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            frame.to_csv(source, index=False)
            train_months = tuple(ALL_MONTHS[:9])
            train_indices = np.flatnonzero(frame["LNMON"].isin(train_months))
            schema = fit_schema(frame, train_indices)
            model = GaussianNB(var_smoothing=1e-9).fit(
                transform_batch(frame.iloc[train_indices], schema),
                frame["TARGET"].to_numpy()[train_indices],
            )
            model_path = root / "v2_model.pkl"
            save_final_model(model_path, model, schema)
            settings = {
                "workflow_version": WORKFLOW_VERSION,
                "experiment": "v2",
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": file_sha256(source),
                "train_months": train_months,
                "validation_months": list(ALL_MONTHS[9:]),
                "validation_month_shares": {202403: 0.6, 202404: 0.3, 202405: 0.1},
                "train_rows": len(train_indices),
                "feature_columns": schema["feature_columns"],
                "hyperparameters": {"decision_threshold": 0.5},
                "history_metric": "misclassification_rate",
                "history_is_weighted": True,
            }
            (root / "v2_hyperparameters.json").write_text(json.dumps(settings), encoding="utf-8")
            pd.DataFrame([{"update": 1, "train_rows": len(train_indices), "valid_error": 0.2}]).to_csv(
                root / "v2_history.csv", index=False
            )
            common = {
                "model": {"path": str(model_path), "settings_path": None, "history_path": None},
                "data": {"path": None, "chunk_rows": 40},
                "output": {"base_dir": str(root / "reports")},
                "runtime": {"prediction_batch_rows": 10},
                "protocols": {
                    "v1": {"validation_months": [202404, 202405], "validation_month_shares": None},
                },
            }
            native = run_validation(OmegaConf.create({**common, "validation": {"protocol": "native"}}))
            summary = pd.read_csv(native / "metrics_summary.csv")
            self.assertIn("overall_weighted", summary["scope"].tolist())
            self.assertIn("common_v1_unweighted", summary["scope"].tolist())
            self.assertTrue((native / "report.html").is_file())
            self.assertTrue((native / "training_curve.png").is_file())
            on_v1 = run_validation(OmegaConf.create({**common, "validation": {"protocol": "v1"}}))
            selected = pd.read_csv(on_v1 / "metrics_summary.csv")
            native_common = summary.loc[summary["scope"].eq("common_v1_unweighted"), "misclassification_rate"].iloc[0]
            v1_error = selected.loc[selected["scope"].eq("overall_unweighted"), "misclassification_rate"].iloc[0]
            self.assertAlmostEqual(native_common, v1_error)
            self.assertEqual(pd.read_csv(on_v1 / "predictions.csv")["LNMON"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
