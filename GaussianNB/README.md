# GaussianNB

This directory contains the Gaussian naive Bayes workflow for binary `TARGET`.
The root CSV is read-only. Every non-`TARGET` column, including `LNMON`, is a
predictor. The shared chronological protocols are in `../docs/ARCHITECTURE.md`.
There is no independent test set.

## Setup and commands

Run all commands from the project root with the workspace Python:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m pip install -r GaussianNB/requirements.txt
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m GaussianNB.train
```

Training creates `GaussianNB/outputs/run_YYYYMMDD_HHMMSS/`. For each protocol,
the final top-level files are `v1_model.pkl`, `v1_hyperparameters.json`, and
`v1_history.csv`, with corresponding `v2_*` files. Checkpoints are stored in
`checkpoints/v1/update_XXXXXX/` and `checkpoints/v2/update_XXXXXX/`.

Run validation with the actual path printed by training:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m GaussianNB.valid model.path=GaussianNB/outputs/RUN_DIR/v1_model.pkl
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m GaussianNB.valid model.path=GaussianNB/outputs/RUN_DIR/v2_model.pkl
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m GaussianNB.valid model.path=GaussianNB/outputs/RUN_DIR/v2_model.pkl validation.protocol=v1
```

To resume one model, select its protocol and a checkpoint directory:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m GaussianNB.train 'runtime.models=[v2]' checkpoint.resume_from=GaussianNB/outputs/RUN_DIR/checkpoints/v2/update_000005
```

Hydra settings in `conf/train.yaml` and `conf/valid.yaml` also accept command
line overrides. Resuming requires the same source CSV, batch size, smoothing
candidates, threshold, and scikit-learn version. It creates a new run directory
and retains earlier outputs.

The default `training.batch_rows=2000` yields approximately 67 v1 updates and
61 v2 updates on the current CSV. Each update consumes a new set of rows once.
The `checkpoint.every=5` default stores roughly every 10,000 processed rows.
The previous 10,000-row run remains valid; changing this setting requires a
new fit and a checkpoint created with the same batch size for resume.

## Method and interpretation

Numeric missing values use medians from the training period; all-missing
columns use zero. Numeric values are standardized with training-period means
and standard deviations, and columns missing in training receive a missingness
indicator. Text columns use training-period one-hot levels, with one level for
missing or unseen values. Each tuning candidate is trained on the earlier
training months and scored by 0/1 error on the configured final training
month(s). The chosen `var_smoothing` is then refit across the full training period with incremental
updates. Validation rows never choose a hyperparameter or checkpoint.

The primary score is misclassification rate at the saved 0.5 threshold. v1
uses equal row weights. v2 gives March, April, and May 60%, 30%, and 10% of
the total validation weight. `valid.py` reports both weighted and unweighted
scores, per-month scores, and a common unweighted April-May score for comparing
the two saved models on identical rows. The zero-only classifier error is also
shown because the source data is undersampled. Probabilities should be read as
scores on this sample, not population event probabilities.

The training-history CSV records processed rows and validation error after
each update. Training can display a live plot on an interactive Matplotlib
backend. `valid.py` reconstructs the training curve and writes a local HTML
report, tables, and plots under a separate `validation/` run directory.

GaussianNB estimates class means and variances from the training data; it does
not directly minimize 0/1 error. The inner smoothing selection and final
report use 0/1 error. Batch size and row order are part of the numerical
method because `partial_fit` applies variance smoothing during each update.
