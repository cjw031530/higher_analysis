# AdaBoost numerical provenance

## Workflow version 1.0.0 — 2026-10-06

- Algorithm: binary discrete AdaBoost-SAMME with depth-3 scikit-learn
  `DecisionTreeClassifier` weak learners, initial learning rate 0.05, and up
  to 300 trees. Tree-level weighted classification errors determine SAMME
  weights. Ensemble validation uses 0/1 misclassification at probability
  threshold 0.5. Version 2 normalizes per-row validation weights to give
  202403, 202404, and 202405 shares of 60%, 30%, and 10%.
- Scientific behavior: train-only one-hot categories, missing/unknown text
  indicator, native dense-tree numeric NaN handling, fixed last-iteration
  model selection, and no validation weights in fitting.
- Runtime versions measured: Python 3.12, NumPy 2.5.1, pandas 3.0.3,
  scikit-learn 1.9.1, Hydra 1.3.4, Matplotlib 3.11.1, tqdm 4.69.0.
- Regression comparison: on a fixed 240-row, eight-feature training sample
  and 80-row validation sample, all 12 fitted SAMME tree weights, tree errors,
  staged validation errors, and final probabilities agreed with the installed
  `sklearn.ensemble.AdaBoostClassifier` within 1e-12. The public tree API's
  fast prediction matched `DecisionTreeClassifier.predict` with NaNs. A
  five-tree checkpoint continuation matched an uninterrupted 12-tree fit
  exactly for history, tree weights, and probabilities.
- Actual-data timing sample: the 162,355-row CSV loaded in 5.355 seconds and
  its 571-column float32 matrix encoded in 0.469 seconds for v1. The first
  three depth-3 boosting rounds on the 133,676-row v1 training portion took
  5.842, 5.904, and 5.845 seconds, including tree fitting, train/validation
  prediction, weight update, and validation scoring. Peak process RSS was
  1,612 MiB, including the loaded DataFrame, encoded matrix, and split
  matrices. This was a single-process sample with BLAS and OpenMP thread
  counts set to one; a two-process full run needs additional memory and may
  have different per-round timings.
- The parity and continuation checks run with `python -m unittest discover
  -s AdaBoost/tests -v`. The timing sample used the actual CSV read-only,
  encoded all predictors with `AdaBoost.data.encode_features`, and called
  `AdaBoost.samme.boost_one` three times on the v1 split.
