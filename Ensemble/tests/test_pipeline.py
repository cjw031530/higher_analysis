"""Small synthetic checks for split isolation, ensemble scores, and replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from Ensemble.data import ALL_MONTHS, fit_schema, transform_features, validation_weights
from Ensemble.model import (
    FeatureSet, average_probabilities, classification_error,
    load_checkpoint, load_models, predict_members,
)
from Ensemble.train import run_training, select_rounds
from Ensemble.valid import run_validation


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for month_index, month in enumerate(ALL_MONTHS):
        for row_index in range(24):
            target = (row_index + (month_index % 3 == 0)) % 2
            rows.append({
                "LNMON": month,
                "TARGET": target,
                "NUMERIC": np.nan if row_index == 0 else target * 3.0 + row_index / 24,
                "CATEGORY": "later" if month >= 202403 and row_index == 0 else ("yes" if target else "no"),
            })
    return pd.DataFrame(rows)


def config(source: Path, output: Path) -> OmegaConf:
    return OmegaConf.create({
        "data": {"path": str(source), "chunk_rows": 51},
        "output": {"base_dir": str(output)},
        "model": {
            "max_rounds": 4, "selection_every": 4, "decision_threshold": 0.5,
            "catboost": {"depth": 2, "learning_rate": 0.1, "random_seed": 17},
            "lightgbm": {"num_leaves": 5, "min_data_in_leaf": 4,
                         "learning_rate": 0.1, "verbosity": -1, "seed": 17},
            "xgboost": {"max_depth": 2, "learning_rate": 0.1,
                        "tree_method": "hist", "random_state": 17},
        },
        "training": {"inner_holdout_months": 1},
        "runtime": {"models": ["v1", "v2"], "parallel_members": True,
                    "threads_per_member": 1, "live_plot": "off"},
        "checkpoint": {"every": 2, "resume_from": None},
        "protocols": {
            "v1": {"train_months": list(ALL_MONTHS[:10]),
                   "validation_months": list(ALL_MONTHS[10:]), "validation_month_shares": None},
            "v2": {"train_months": list(ALL_MONTHS[:9]),
                   "validation_months": list(ALL_MONTHS[9:]),
                   "validation_month_shares": {202403: 0.6, 202404: 0.3, 202405: 0.1}},
        },
    })


def validation_config(source: Path, model_path: Path, output: Path,
                      protocol: str = "native", comparison: Path | None = None):
    return OmegaConf.create({
        "model": {"path": str(model_path), "settings_path": None, "history_path": None},
        "comparison_model": {"path": None if comparison is None else str(comparison)},
        "data": {"path": str(source), "chunk_rows": 57},
        "validation": {"protocol": protocol},
        "output": {"base_dir": str(output)},
        "runtime": {"threads_per_member": 1},
        "protocols": config(source, output).protocols,
    })


class EnsemblePipelineTests(unittest.TestCase):
    def test_weights_schema_and_zero_one_loss(self) -> None:
        frame = synthetic_frame()
        schema = fit_schema(frame.loc[frame["LNMON"] <= 202402])
        self.assertEqual(schema["feature_columns"], ["LNMON", "NUMERIC", "CATEGORY"])
        self.assertNotIn("later", schema["categorical_levels"]["CATEGORY"])
        _, encoded = transform_features(frame.loc[frame["LNMON"] == 202403], schema)
        self.assertTrue(pd.isna(encoded.iloc[0]["CATEGORY"]))
        months = np.array([202403, 202403, 202404, 202405, 202405])
        weights = validation_weights(months, {202403: 0.6, 202404: 0.3, 202405: 0.1})
        np.testing.assert_allclose([weights[:2].sum(), weights[2], weights[3:].sum()], [0.6, 0.3, 0.1])
        scores = {name: np.array([0.6, 0.2, 0.7]) for name in ("catboost", "lightgbm", "xgboost")}
        np.testing.assert_allclose(average_probabilities(scores), [0.6, 0.2, 0.7])
        self.assertAlmostEqual(classification_error(np.array([1, 0, 1]), np.array([0.5, 0.1, 0.9])), 1 / 3)
        cfg = config(Path("unused.csv"), Path("unused_outputs"))
        params = {name: OmegaConf.to_container(cfg.model[name]) for name in ("catboost", "lightgbm", "xgboost")}
        selected, inner = select_rounds(frame, tuple(ALL_MONTHS[:9]), 1, 4, 2,
                                        params, 1, True, 0.5, "synthetic")
        self.assertIn(selected, (2, 4))
        self.assertEqual([row["round"] for row in inner], [2, 4])

    def test_train_validate_cross_protocol_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            synthetic_frame().to_csv(source, index=False)
            cfg = config(source, root / "outputs")
            run_dir = run_training(cfg)
            v1 = run_dir / "v1_model"
            v2 = run_dir / "v2_model"
            self.assertTrue((v1 / "catboost.cbm").is_file())
            self.assertTrue((v2 / "xgboost.json").is_file())
            v2_settings = json.loads((run_dir / "v2_hyperparameters.json").read_text())
            self.assertEqual(v2_settings["train_months"], list(ALL_MONTHS[:9]))
            self.assertEqual(v2_settings["validation_months"], list(ALL_MONTHS[9:]))
            self.assertEqual(v2_settings["selected_rounds"], 4)
            self.assertEqual(v2_settings["inner_selection_history"][0]["round"], 4)
            self.assertTrue((run_dir / "checkpoints" / "v2" / "round_000002" / "state.json").is_file())

            native = run_validation(validation_config(source, v2, root / "reports"))
            summary = pd.read_csv(native / "metrics_summary.csv")
            self.assertIn("overall_weighted", summary["scope"].tolist())
            self.assertIn("common_v1_unweighted", summary["scope"].tolist())
            self.assertTrue((native / "report.html").is_file())
            self.assertTrue((native / "training_curve.png").is_file())
            on_v1 = run_validation(validation_config(source, v2, root / "reports", "v1", v1))
            selected = pd.read_csv(on_v1 / "metrics_summary.csv")
            native_common = summary.loc[summary["scope"].eq("common_v1_unweighted"), "misclassification_rate"].iloc[0]
            v1_window = selected.loc[selected["scope"].eq("overall_unweighted"), "misclassification_rate"].iloc[0]
            self.assertAlmostEqual(native_common, v1_window)
            self.assertIn("v1_comparison_ensemble", pd.read_csv(on_v1 / "model_comparison.csv")["model"].tolist())
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_validation(validation_config(source, v1, root / "reports", "v2"))

            checkpoint = run_dir / "checkpoints" / "v2" / "round_000002"
            state, _, history = load_checkpoint(checkpoint, 1)
            self.assertEqual(state["completed_rounds"], 2)
            self.assertEqual(len(history), 1)
            cfg.runtime.models = ["v2"]
            cfg.checkpoint.resume_from = str(checkpoint)
            resumed = run_training(cfg)
            expected_history = pd.read_csv(run_dir / "v2_history.csv")
            actual_history = pd.read_csv(resumed / "v2_history.csv")
            self.assertEqual(actual_history["round"].tolist(), expected_history["round"].tolist())
            np.testing.assert_allclose(actual_history["valid_error"], expected_history["valid_error"], atol=1e-12)

            changed = synthetic_frame()
            outer = changed["LNMON"].isin(ALL_MONTHS[9:])
            changed.loc[outer, "TARGET"] = 1 - changed.loc[outer, "TARGET"]
            changed_source = root / "changed_outer_labels.csv"
            changed.to_csv(changed_source, index=False)
            changed_cfg = config(changed_source, root / "changed_outputs")
            changed_cfg.runtime.models = ["v2"]
            changed_run = run_training(changed_cfg)
            changed_settings = json.loads((changed_run / "v2_hyperparameters.json").read_text())
            self.assertEqual(changed_settings["selected_rounds"], v2_settings["selected_rounds"])
            self.assertEqual(changed_settings["inner_selection_history"], v2_settings["inner_selection_history"])
            common_features = synthetic_frame().loc[lambda x: x["LNMON"].isin(ALL_MONTHS[9:])]
            native, encoded = transform_features(common_features, v2_settings["schema"])
            features = FeatureSet(native, encoded, None, v2_settings["schema"]["categorical_features"])
            original_scores = predict_members(load_models(v2, 1), features, 1)
            changed_scores = predict_members(load_models(changed_run / "v2_model", 1), features, 1)
            for member in original_scores:
                np.testing.assert_allclose(changed_scores[member], original_scores[member], atol=1e-12)
            self.assertNotEqual(
                pd.read_csv(changed_run / "v2_history.csv")["valid_error"].iloc[-1],
                expected_history["valid_error"].iloc[-1],
            )


if __name__ == "__main__":
    unittest.main()
