# Ensemble

This directory trains an equal-probability average of CatBoost, LightGBM, and
XGBoost for the binary `TARGET` task. The root CSV is read-only. All columns
except `TARGET`, including `LNMON`, are predictors. No independent test set is
available. The split definitions and primary metric are in
`../docs/ARCHITECTURE.md`.

## Setup and commands

Run these commands from the project root using the designated virtual
environment:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m pip install -r Ensemble/requirements.txt
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.train
```

Training creates `Ensemble/outputs/run_YYYYMMDD_HHMMSS/`. Each protocol has
`v1_model/` or `v2_model/` with the three native model files, a matching
`*_hyperparameters.json`, and a `*_history.csv`. The only other training
artifacts are complete resume checkpoints in `checkpoints/PROTOCOL/round_N/`.
No plots or row-level predictions are saved during training. In an interactive
Matplotlib session, the outer validation curve is displayed live. In a
headless session, `tqdm` displays progress and `valid.py` later renders the
saved curve.

Validate the saved models after training, replacing `RUN_DIR` with the printed
directory name:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.valid model.path=Ensemble/outputs/RUN_DIR/v1_model
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.valid model.path=Ensemble/outputs/RUN_DIR/v2_model
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.valid model.path=Ensemble/outputs/RUN_DIR/v2_model validation.protocol=v1 comparison_model.path=Ensemble/outputs/RUN_DIR/v1_model
```

The third command evaluates both saved models on the same April-May rows. It
is the direct comparison for separating model changes from v2's March-heavy
weighted validation score. `valid.py` writes CSV tables, PNG plots, metadata,
and `report.html` in a separate `validation/` directory. Its primary score
and `primary_loss` are the same 0/1 misclassification rate.

Resume one protocol from a complete checkpoint in a new run directory:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.train 'runtime.models=[v2]' checkpoint.resume_from=Ensemble/outputs/RUN_DIR/checkpoints/v2/round_000020
```

Hydra settings in `conf/train.yaml`, `conf/valid.yaml`, and
`conf/protocols/default.yaml` accept command-line overrides. A resume checks
the source hash, feature schema, split, hyperparameters, thread count, and
library versions. The source and model files must remain available.

## Training and interpretation

The final threshold is fixed at 0.5: probabilities strictly above 0.5 predict
class 1. The ensemble probability is the arithmetic mean of the three class-1
probabilities. CatBoost, LightGBM, and XGBoost each optimize a differentiable
binary logistic loss. Zero-one loss chooses a common boosting-round count on
the final month of the *training* period: March 2024 for v1 and February 2024
for v2. The selected models are then fitted again on their full training
months. A tie in inner zero-one loss selects fewer rounds.

Outer validation labels never enter the fit, feature schema, round choice, or
checkpoint choice. During final fitting they are scored only to display the
curve and record its underlying values. v1 validation uses equal row weights;
v2 gives March, April, and May 60%, 30%, and 10% of the *validation* metric's
total weight. These are not training sample weights. `valid.py` also reports
unweighted and monthly scores, the common April-May window, each member, and
an always-zero baseline. The source is undersampled, so probabilities describe
scores on this sample, not population event rates.

Categorical levels for LightGBM and XGBoost are learned only from the relevant
training period; unseen validation categories become missing. CatBoost uses
native categorical inputs and a reserved missing-category string. Numeric
missing values remain missing for each tree library. All predictors are kept.
Both inner and final training advance in blocks, with members fitted in
parallel and native library threads capped per member. Each completed final
block writes an atomic checkpoint with all three models and history. The
checkpoint interval also determines the resolution of the outer validation
curve and can affect segmented learning results; changing it requires a new
run. See `NUMERICAL_PROVENANCE.md` for the synthetic comparison and benchmark.

The repository's full CSV has **not** been trained or validated by the
implementation workflow. Run the commands above when ready.
