"""Check full-state SAGA continuation and scikit-learn objective agreement."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ElasticNet.saga import (
    initial_state, load_checkpoint, predict_probability, run_epoch, save_checkpoint,
)


class SAGAStateTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(19)
        self.X = np.ascontiguousarray(rng.normal(size=(320, 9)), dtype="float32")
        probability = expit(self.X[:, 0] * 0.9 - self.X[:, 1] * 0.6)
        self.y = (rng.random(len(self.X)) < probability).astype("float32")

    def test_full_checkpoint_matches_uninterrupted_epoch_schedule(self) -> None:
        full = initial_state(self.X, 0.1, 0.5, 42)
        for _ in range(12):
            run_epoch(full, self.X, self.y, 0.1, 0.5)
        resumed = initial_state(self.X, 0.1, 0.5, 42)
        history = []
        for epoch in range(5):
            change = run_epoch(resumed, self.X, self.y, 0.1, 0.5)
            history.append({"epoch": epoch + 1, "valid_error": 0.2, "relative_coef_change": change})
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(Path(temporary), resumed, {"test": True}, history)
            settings, resumed, loaded_history = load_checkpoint(path)
            self.assertEqual(settings, {"test": True})
            self.assertEqual(len(loaded_history), 5)
            for _ in range(7):
                run_epoch(resumed, self.X, self.y, 0.1, 0.5)
        for name in (
            "coef", "intercept", "sum_gradient", "intercept_sum_gradient",
            "gradient_memory", "seen",
        ):
            np.testing.assert_array_equal(getattr(full, name), getattr(resumed, name))
        self.assertEqual((full.num_seen, full.epoch), (resumed.num_seen, resumed.epoch))

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        state = initial_state(self.X, 0.1, 0.5, 42)
        change = run_epoch(state, self.X, self.y, 0.1, 0.5)
        history = [{"epoch": 1, "valid_error": 0.2, "relative_coef_change": change}]
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(Path(temporary), state, {"test": True}, history)
            with (path / "optimizer_state.npz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(path)

    def test_converged_solution_agrees_with_sklearn_logistic_regression(self) -> None:
        state = initial_state(self.X, 0.1, 0.5, 42)
        for _ in range(200):
            run_epoch(state, self.X, self.y, 0.1, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            reference = LogisticRegression(
                solver="saga", penalty="elasticnet", l1_ratio=0.5, C=0.1,
                max_iter=200, tol=0, random_state=42,
            ).fit(self.X, self.y)
        actual = predict_probability(self.X, state.coef, state.intercept)
        expected = reference.predict_proba(self.X)[:, 1]
        np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0)


if __name__ == "__main__":
    unittest.main()
