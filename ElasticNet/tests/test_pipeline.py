"""Exercise training, exact checkpoint replay, and cross-protocol validation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ElasticNet.saga import load_final_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", *arguments],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def test_cli_train_resume_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            months = list(range(202306, 202313)) + list(range(202401, 202406))
            rows = []
            for month_index, month in enumerate(months):
                for item in range(16):
                    rows.append({
                        "LNMON": month,
                        "TARGET": int(item % 5 == 0 or (month_index > 8 and item == 1)),
                        "numeric": float(item + month_index / 5),
                        "missing_numeric": np.nan if item % 3 == 0 else float(item % 4),
                        "all_missing_early": np.nan if month_index < 9 else float(item),
                        "text": "later" if month_index >= 10 and item == 0 else ("A" if item % 2 else "B"),
                    })
            csv_path = root / "synthetic.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            outputs = root / "outputs"
            common = [
                f"data.path={csv_path}", f"output.base_dir={outputs}",
                "model.max_epochs=3", "model.tol=0", "checkpoint.every=1",
                "runtime.parallel_models=false", "runtime.live_plot=off",
            ]
            self.run_cli("ElasticNet.train", *common)
            original = sorted(outputs.glob("run_*"))[0]
            self.assertTrue((original / "v1_model.pkl").is_file())
            self.assertTrue((original / "v2_model.pkl").is_file())
            checkpoint = original / "checkpoints" / "v2" / "epoch_000001"
            self.run_cli("ElasticNet.train", *common, f"checkpoint.resume_from={checkpoint}")
            resumed_run = sorted(outputs.glob("run_*"))[1]
            original_model = load_final_model(original / "v2_model.pkl")
            resumed_model = load_final_model(resumed_run / "v2_model.pkl")
            np.testing.assert_array_equal(original_model["coef"], resumed_model["coef"])
            self.assertEqual(original_model["intercept"], resumed_model["intercept"])
            pd.testing.assert_frame_equal(
                pd.read_csv(original / "v2_history.csv"),
                pd.read_csv(resumed_run / "v2_history.csv"),
            )
            valid_model = original / "v2_model.pkl"
            self.run_cli("ElasticNet.valid", f"model.path={valid_model}")
            self.run_cli("ElasticNet.valid", f"model.path={valid_model}", "validation.protocol=v1")
            native = sorted((original / "validation").glob("v2_on_v2_*"))[0]
            cross = sorted((original / "validation").glob("v2_on_v1_*"))[0]
            self.assertTrue((native / "report.html").is_file())
            self.assertTrue((cross / "training_curve.png").is_file())
            predictions = pd.read_csv(cross / "predictions.csv")
            self.assertEqual(set(predictions["LNMON"]), {202404, 202405})
            summary = pd.read_csv(cross / "metrics_summary.csv")
            score = float(summary.loc[summary["scope"].eq("overall_unweighted"), "misclassification_rate"].iloc[0])
            self.assertAlmostEqual(score, float((predictions["predicted_target"] != predictions["TARGET"]).mean()))
            metadata = json.loads((native / "validation_metadata.json").read_text())
            self.assertEqual(metadata["validation_month_shares"], {"202403": 0.6, "202404": 0.3, "202405": 0.1})
            native_predictions = pd.read_csv(native / "predictions.csv")
            monthly_errors = native_predictions.groupby("LNMON").apply(
                lambda group: (group["predicted_target"] != group["TARGET"]).mean(),
                include_groups=False,
            )
            expected_weighted = (
                0.6 * monthly_errors.loc[202403]
                + 0.3 * monthly_errors.loc[202404]
                + 0.1 * monthly_errors.loc[202405]
            )
            self.assertAlmostEqual(metadata["primary_score"], float(expected_weighted))


if __name__ == "__main__":
    unittest.main()
