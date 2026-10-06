# Decisions

## 2026-10-06: Chronological baseline and validation weighting

Use the exact two requested month splits. Keep `LNMON` as a predictor because
the input is explicitly defined as every non-target column. Apply version 2's
60/30/10 weights to validation only, normalizing by the number of rows in each
month so row count does not change the intended month shares.

## 2026-10-06: Initial model and categories

Use XGBoost's histogram tree method and native categorical support with the
requested 1,600-tree hyperparameters. Preserve all predictors for the initial
baseline, including sparse and constant columns. Derive each model's category
vocabulary only from its training period and serialize models as JSON.

## 2026-10-06: Execution and dependencies

Run the two models concurrently with a separate CPU thread budget per model.
Use tqdm to expose data loading and training progress. Keep dependency
declarations in `XGBoost/requirements.txt`; the root requirements file and
root data artifacts are not changed. Use XGBoost 3.2 or newer for pandas 3
compatibility and categorical support, together with scikit-learn for metrics.
