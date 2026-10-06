# KCB XGBoost baseline

This module trains two binary XGBoost classifiers from the read-only CSV at the
project root. Every column except `TARGET`, including `LNMON`, is a predictor.
It does not create a test split.

## Setup

Use the workspace Python environment specified in the project instructions:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m pip install -r XGBoost/requirements.txt
```

## Run

From the project root:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.train
```

The default run uses 1,600 trees per model and trains the two models concurrently.
Each model receives half of the available CPU threads. A progress bar shows CSV
rows and boosting iterations. Use `--jobs-per-model N` to set the thread count
per model, or `--sequential` to reduce peak memory. `--n-estimators N`, `--data
PATH`, and `--output-base PATH` are available for controlled runs and smoke tests.

Each run creates a new directory under `XGBoost/outputs/` containing:

- `run.json`: data source, feature schema, package versions, and runtime settings.
- `v1_model.json` and `v2_model.json`: category-preserving XGBoost models.
- `v1_metrics.json` and `v2_metrics.json`: split details, overall metrics, and
  monthly unweighted metrics.
- `v1_validation_predictions.csv` and `v2_validation_predictions.csv`: one
  probability per validation row with its original zero-based CSV row index.

Version 1 trains on 202306–202403 and validates on 202404–202405. Version 2
trains on 202306–202402 and validates on 202403–202405. In version 2, each row
receives its month's desired share divided by that month's validation row count.
This makes the total validation weight 60% for 202403, 30% for 202404, and 10%
for 202405. These weights affect validation metrics only.

Both models use `tree_method="hist"`, `enable_categorical=True`,
`n_estimators=1600`, `max_depth=5`, `learning_rate=0.04`,
`min_child_weight=10`, `subsample=0.8`, and `colsample_bytree=0.8` by default.
No early stopping or hyperparameter search is applied. ROC AUC, average
precision, and log loss are reported. The validation scores are not independent
test scores, and observed probabilities reflect the supplied undersampled data.
