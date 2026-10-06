"""Regression checks for SAMME parity and exact checkpoint continuation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from AdaBoost import WORKFLOW_VERSION
from AdaBoost.samme import (
    SAMMEModel, boost_one, initial_snapshot, load_checkpoint, save_checkpoint,
    tree_predict,
)


class SAMMERegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(52)
        self.train_x = rng.normal(size=(240, 8)).astype("float32")
        self.valid_x = rng.normal(size=(80, 8)).astype("float32")
        latent = self.train_x[:, 0] + 0.4 * self.train_x[:, 1] * self.train_x[:, 2]
        self.train_y = ((latent + rng.normal(scale=1.2, size=len(latent))) > 0.4).astype("int8")
        valid_latent = self.valid_x[:, 0] + 0.4 * self.valid_x[:, 1] * self.valid_x[:, 2]
        self.valid_y = ((valid_latent + rng.normal(scale=1.2, size=len(valid_latent))) > 0.4).astype("int8")
        self.parameters = dict(max_depth=3, n_estimators=12, learning_rate=0.05, random_state=42)

    def fit_custom(self, rounds: int = 12, snapshot=None):
        snapshot = snapshot or initial_snapshot(SAMMEModel(**self.parameters), len(self.train_y), len(self.valid_y))
        while snapshot.model.fitted_iterations < rounds and snapshot.stopping_reason is None:
            boost_one(snapshot, self.train_x, self.train_y, self.valid_x, self.valid_y, None)
        return snapshot

    def test_matches_sklearn_staged_predictions_and_probabilities(self) -> None:
        reference = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=12, learning_rate=0.05, random_state=42,
        ).fit(self.train_x, self.train_y)
        snapshot = self.fit_custom()
        self.assertEqual(snapshot.model.fitted_iterations, len(reference.estimators_))
        np.testing.assert_allclose(snapshot.model.estimator_weights, reference.estimator_weights_[:12], rtol=1e-12)
        np.testing.assert_allclose(snapshot.model.estimator_errors, reference.estimator_errors_[:12], rtol=1e-12)
        reference_history = [np.mean(pred != self.valid_y) for pred in reference.staged_predict(self.valid_x)]
        np.testing.assert_allclose(snapshot.history, reference_history, rtol=0, atol=0)
        np.testing.assert_allclose(
            snapshot.model.predict_proba(self.valid_x), reference.predict_proba(self.valid_x), rtol=1e-12, atol=1e-12,
        )

    def test_checkpoint_resume_matches_uninterrupted_fit(self) -> None:
        uninterrupted = self.fit_custom()
        halfway = self.fit_custom(rounds=5)
        with tempfile.TemporaryDirectory() as temporary:
            destination = save_checkpoint(Path(temporary), halfway, {"workflow_version": WORKFLOW_VERSION})
            _, restored = load_checkpoint(destination)
            resumed = self.fit_custom(snapshot=restored)
        np.testing.assert_allclose(resumed.history, uninterrupted.history, rtol=0, atol=0)
        np.testing.assert_allclose(resumed.model.estimator_weights, uninterrupted.model.estimator_weights, rtol=0, atol=0)
        np.testing.assert_allclose(
            resumed.model.predict_proba(self.valid_x), uninterrupted.model.predict_proba(self.valid_x), rtol=0, atol=0,
        )

    def test_fast_tree_prediction_handles_nan(self) -> None:
        features = self.train_x.copy()
        features[::7, 2] = np.nan
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(features, self.train_y)
        np.testing.assert_array_equal(tree_predict(tree, features), tree.predict(features))


if __name__ == "__main__":
    unittest.main()
