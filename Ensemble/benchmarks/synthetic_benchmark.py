"""Benchmark and compare continuous and checkpoint-sized native training."""

from __future__ import annotations

import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from Ensemble.data import fit_schema, transform_features
from Ensemble.model import (
    FeatureSet, average_probabilities, classification_error,
    fit_members, predict_members,
)


def main() -> None:
    rng = np.random.default_rng(20261006)
    rows = 8000
    numeric_columns = 64
    numeric = rng.normal(size=(rows, numeric_columns)).astype("float32")
    numeric[rng.random(size=numeric.shape) < 0.04] = np.nan
    frame = pd.DataFrame(numeric, columns=[f"F{index:03d}" for index in range(numeric_columns)])
    frame.insert(0, "LNMON", np.repeat([202301, 202302], rows // 2))
    frame["CATEGORY"] = rng.choice(["A", "B", "C"], size=rows)
    signal = np.nan_to_num(numeric[:, 0]) + 0.5 * np.nan_to_num(numeric[:, 1])
    frame["TARGET"] = (signal + 0.2 * rng.normal(size=rows) > 0).astype("int8")
    schema = fit_schema(frame.iloc[:6000])
    train_native, train_encoded = transform_features(frame.iloc[:6000], schema)
    valid_native, valid_encoded = transform_features(frame.iloc[6000:], schema)
    train = FeatureSet(train_native, train_encoded, frame["TARGET"].to_numpy()[:6000], schema["categorical_features"])
    valid = FeatureSet(valid_native, valid_encoded, frame["TARGET"].to_numpy()[6000:], schema["categorical_features"])
    params = {
        "catboost": {"depth": 4, "learning_rate": 0.05, "random_seed": 42},
        "lightgbm": {"num_leaves": 15, "min_data_in_leaf": 20,
                     "learning_rate": 0.05, "verbosity": -1, "seed": 42},
        "xgboost": {"max_depth": 4, "learning_rate": 0.05,
                    "tree_method": "hist", "random_state": 42},
    }
    start = perf_counter()
    continuous = fit_members(train, 20, params, threads=2)
    continuous_seconds = perf_counter() - start
    start = perf_counter()
    first = fit_members(train, 10, params, threads=2)
    segmented = fit_members(train, 10, params, threads=2, previous=first)
    segmented_seconds = perf_counter() - start
    continuous_scores = predict_members(continuous, valid, threads=2)
    segmented_scores = predict_members(segmented, valid, threads=2)
    result = {
        "rows": rows, "numeric_columns": numeric_columns,
        "train_rows": 6000, "validation_rows": 2000,
        "continuous_seconds": round(continuous_seconds, 4),
        "segmented_seconds": round(segmented_seconds, 4),
        "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "member_max_absolute_probability_difference": {
            name: float(np.max(np.abs(continuous_scores[name] - segmented_scores[name])))
            for name in continuous_scores
        },
        "continuous_ensemble_error": classification_error(
            valid.target, average_probabilities(continuous_scores)),
        "segmented_ensemble_error": classification_error(
            valid.target, average_probabilities(segmented_scores)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
