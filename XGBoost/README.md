# KCB XGBoost baseline

This module trains two binary XGBoost classifiers from the read-only monthly
CSV at the project root. Every column except `TARGET`, including `LNMON`, is a
predictor. No independent test split is available.

## Setup

Use the workspace Python environment specified in the project instructions:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m pip install -r XGBoost/requirements.txt
```

## Train

Edit `conf/train.yaml` for data/output paths, model hyperparameters, checkpoint
frequency, and runtime settings. The two month splits and validation month
shares are in `conf/protocols/default.yaml`. Hydra command-line overrides are
also supported. Run from the project root:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.train
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.train model.max_depth=6 checkpoint.every=50
```

Version 1 trains on 202306–202403 and validates on 202404–202405. Version 2
trains on 202306–202402 and validates on 202403–202405. Version 2 divides each
month's desired share by its validation row count, giving total validation
weights of 60%, 30%, and 10% for 202403, 202404, and 202405. The weights affect
evaluation only.

Both models use `tree_method="hist"`, `enable_categorical=True`,
`n_estimators=3000`, `max_depth=5`, `learning_rate=0.04`,
`min_child_weight=10`, `subsample=0.8`, and `colsample_bytree=0.8` by default.
There is no early stopping or parameter search. `binary:logistic` remains the
fit objective because 0/1 misclassification loss has no useful gradient for
XGBoost's tree updates. `eval_metric=error` monitors misclassification at a
0.5 probability cutoff; it does not change how the trees are fitted.

Training runs the models concurrently, assigning half the available CPU
threads to each by default. tqdm reports CSV loading and boosting iterations,
including current validation misclassification rate. With an interactive Matplotlib backend,
a live plot updates both validation-error curves every 25 iterations. With a
noninteractive backend, the full curve remains available from the saved
history. Set `runtime.live_plot=on` to require a GUI or `off` to disable the
window. Set `runtime.parallel_models=false` to reduce peak memory and
`runtime.jobs_per_model=N` to control CPU threads.

Each invocation creates a unique directory under `XGBoost/outputs/`. Training
saves three final files for each version:

- `v1_model.json` / `v2_model.json`: final category-preserving model.
- `v1_hyperparameters.json` / `v2_hyperparameters.json`: hyperparameters plus
  the split, feature schema, training categories, source, and package versions
  required for reproducible validation.
- `v1_history.csv` / `v2_history.csv`: iteration and validation error rate, with
  all iterations retained. Version 2's error uses its month weights.

`checkpoints/v1/round_000100/` and `checkpoints/v2/round_000100/` contain the
periodic snapshots when `checkpoint.every=100`. Each checkpoint directory has
`model.pkl`, `history.csv`, and `state.json`. The snapshot retains XGBoost's
training state and is intended for continuation with the same XGBoost version.
The final JSON models are the portable models for validation.

Resume a selected model from a checkpoint directory. A new run directory is
created, leaving the earlier checkpoint and run untouched. The configured
total tree count and fit hyperparameters must match the checkpoint. The command
continues only the model named in the checkpoint:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.train checkpoint.resume_from=XGBoost/outputs/run_YYYYMMDD_HHMMSS/checkpoints/v2/round_000100
```

## Validate a saved model

Edit `conf/valid.yaml` to set the model path, data path, output path, and
validation protocol, or override them on the command line. Pass a model path
produced by training:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.valid model.path=XGBoost/outputs/run_YYYYMMDD_HHMMSS/v1_model.json
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m XGBoost.valid model.path=XGBoost/outputs/run_YYYYMMDD_HHMMSS/v2_model.json validation.protocol=v1
```

`validation.protocol=native` uses the model's original validation definition.
The second example evaluates the v2 model on the v1 validation period
(202404–202405) without month weights. Comparing that output with the v1 model
on its native protocol holds the validation period fixed. A protocol is rejected
if any of its validation months were used to train the model.

The validator reads the adjacent hyperparameters and history files and uses
the original CSV by default. `data.path=PATH` can point to a CSV containing
only the required validation months, provided its predictor schema matches.
Use `model.settings_path=PATH` and `model.history_path=PATH` if the saved files
have been moved.
Validation creates a new subdirectory beside the model, containing `report.html`, predictions,
overall and monthly metrics, a same-model protocol comparison table,
score-decile calibration, threshold diagnostics,
and PNG figures for the training curve, ROC/precision-recall curves,
calibration, monthly metrics, and score distributions. Version 2's overall
weighted metrics and plots use the 60/30/10 month shares. The threshold table
is diagnostic; the final classification score uses a fixed 0.5 cutoff. The
primary score is `misclassification_rate`: the weighted mean of wrong
predictions for v2, and the ordinary fraction wrong for v1. Lower is better.
ROC AUC, average precision, log loss, and Brier score remain supplementary.
Older log-loss training artifacts can still be validated; their protocol
comparison keeps log loss, while the final score is misclassification rate.

Validation results describe the supplied undersampled data and are not
independent test results.
