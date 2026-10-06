"""Exercise both protocols, saved-model validation, and checkpoint resume."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from AdaBoost.samme import load_final_model


ROOT = Path(__file__).resolve().parents[2]
MONTHS = (
    202306, 202307, 202308, 202309, 202310, 202311,
    202312, 202401, 202402, 202403, 202404, 202405,
)


class PipelineTests(unittest.TestCase):
    def run_module(self, module: str, *overrides: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", module, *overrides], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            self.fail(f"{module} failed:\n{result.stdout}\n{result.stderr}")

    def test_parallel_training_resume_and_cross_protocol_validation(self) -> None:
        rng = np.random.RandomState(13)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            months = np.repeat(MONTHS, 20)
            numeric = rng.normal(size=len(months))
            other = rng.normal(size=len(months))
            categories = np.where(numeric > 0, "A", "B").astype(object)
            categories[months == 202404] = "UNSEEN"
            numeric[::9] = np.nan
            likelihood = 1 / (1 + np.exp(-(0.9 * np.nan_to_num(numeric) + 0.3 * other - 1)))
            target = (rng.rand(len(months)) < likelihood).astype("int8")
            source = folder / "sample.csv"
            pd.DataFrame({
                "LNMON": months, "TARGET": target, "NUM": numeric,
                "OTHER": other, "TEXT": categories,
            }).to_csv(source, index=False)
            original_bytes = source.read_bytes()
            output = folder / "outputs"
            self.run_module(
                "AdaBoost.train", f"data.path={source}", f"output.base_dir={output}",
                "model.n_estimators=6", "checkpoint.every=3", "runtime.live_plot=off",
                "data.chunk_rows=50",
            )
            first_run = next(output.iterdir())
            checkpoint = first_run / "checkpoints" / "v2" / "round_000003"
            self.assertTrue(checkpoint.is_dir())
            self.assertFalse(any(first_run.glob(".matrix_*")))
            for version in ("v1", "v2"):
                settings = json.loads((first_run / f"{version}_hyperparameters.json").read_text())
                self.assertEqual(settings["runtime"], {"parallel_backend": "threads", "training_threads": 2})
            # Version 1.0.0 checkpoints contain the same numerical SAMME state.
            state_path = checkpoint / "state.json"
            old_state = json.loads(state_path.read_text())
            old_state["workflow_version"] = "1.0.0"
            state_path.write_text(json.dumps(old_state))
            self.run_module(
                "AdaBoost.train", f"data.path={source}", f"output.base_dir={output}",
                "model.n_estimators=6", f"checkpoint.resume_from={checkpoint}",
                "runtime.live_plot=off", "data.chunk_rows=50",
            )
            second_run = next(path for path in output.iterdir() if path != first_run)
            original = load_final_model(first_run / "v2_model.pkl")
            resumed = load_final_model(second_run / "v2_model.pkl")
            np.testing.assert_array_equal(original.estimator_weights, resumed.estimator_weights)
            pd.testing.assert_frame_equal(
                pd.read_csv(first_run / "v2_history.csv"),
                pd.read_csv(second_run / "v2_history.csv"),
            )
            for version, protocol in (("v1", "native"), ("v2", "native"), ("v2", "v1")):
                self.run_module(
                    "AdaBoost.valid", f"model.path={first_run / (version + '_model.pkl')}",
                    f"validation.protocol={protocol}", "data.chunk_rows=50",
                )
                report = max((first_run / "validation").glob(f"{version}_on_*"), key=lambda path: path.stat().st_mtime_ns)
                self.assertTrue((report / "report.html").is_file())
                self.assertTrue((report / "confusion_matrix.png").is_file())
                metadata = json.loads((report / "validation_metadata.json").read_text())
                summary = pd.read_csv(report / "metrics_summary.csv")
                primary_scope = "overall_weighted" if version == "v2" and protocol == "native" else "overall_unweighted"
                expected = summary.loc[summary["scope"].eq(primary_scope), "misclassification_rate"].iloc[0]
                self.assertAlmostEqual(metadata["primary_score"], expected)
                if protocol == "native":
                    history = pd.read_csv(first_run / f"{version}_history.csv")
                    self.assertAlmostEqual(history["valid_error"].iloc[-1], expected)
            self.assertEqual(source.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
