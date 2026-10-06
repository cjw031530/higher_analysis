# KCB AdaBoost baseline

This package trains binary AdaBoost-SAMME classifiers on the read-only CSV at
the project root. Every column except `TARGET`, including `LNMON`, is a model
input. The EDA notebook describes 162,355 rows and 559 model inputs. No
independent test set is available.

## Setup

Use the Python environment specified for this workspace:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m pip install -r AdaBoost/requirements.txt
```

Dependencies are declared only in `AdaBoost/requirements.txt`. The root
requirements file and CSV are not changed by this workflow.

## Train

Run from the project root:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.train
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.train model.n_estimators=300 checkpoint.every=50
```

Hydra configuration is in `conf/train.yaml` and `conf/protocols/default.yaml`.
The initial model uses a depth-3 decision tree, 300 maximum boosting rounds,
and learning rate 0.05. The final model is always the **last fitted ensemble**;
the lowest validation error and its iteration are recorded for reference only.
Standard AdaBoost can stop before 300 rounds if a weak tree is perfect or no
better than random. Such a stop is recorded in the settings.

| Protocol | Training | Validation | Primary validation error |
| --- | --- | --- | --- |
| v1 | 202306–202403 | 202404–202405 | Mean row misclassification |
| v2 | 202306–202402 | 202403–202405 | 60% / 30% / 10% monthly misclassification |

For v2, each row receives its month's share divided by the number of rows in
that month. These weights affect validation metrics, not model fitting. The
classification cutoff is 0.5, with ties assigned to class 0. The training
history records ensemble misclassification after every tree. SAMME fits each
tree using the current sample weights and increases the weights of that tree's
mistakes. Its optimization is an exponential surrogate; it does not directly
minimize the ensemble's discontinuous 0/1 error.

Four text predictors in the current CSV are one-hot encoded using levels from
the selected training months only. Missing or previously unseen text values
get a separate indicator. Numeric predictors retain missing values, which
`DecisionTreeClassifier` handles in the supported scikit-learn version. The
source CSV, predictor order, category levels, package versions, and numeric
workflow version are recorded with the model.

The two protocols run in separate processes by default. Their read-only
float32 feature matrices are temporary memory-mapped files, removed when the
run finishes. Individual AdaBoost trees are sequential because each depends
on the previous tree's sample weights. `tqdm` displays the current validation
error for each protocol. An interactive Matplotlib backend can show both
curves live; `runtime.live_plot=off` disables the window. `valid.py` generates
the saved training plot from the recorded history.

Each run creates a unique `AdaBoost/outputs/run_.../` directory. The three
final files per protocol are:

- `v1_model.pkl` or `v2_model.pkl`: the last fitted ensemble.
- `v1_hyperparameters.json` or `v2_hyperparameters.json`: configuration,
  preprocessing schema, data provenance, and final fit status.
- `v1_history.csv` or `v2_history.csv`: iteration and validation error.

Checkpoints are separate under `checkpoints/v1/round_000050/` and
`checkpoints/v2/round_000050/`. Each contains `snapshot.pkl`, `history.csv`,
and `state.json`. A checkpoint stores the model, next-round training weights,
random generator state, and current validation margin. Checkpoints and final
pickle models should be loaded only from trusted runs and require the same
scikit-learn version for reliable continuation.

Resume a single protocol into a new run directory:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.train checkpoint.resume_from=AdaBoost/outputs/run_YYYYMMDD_HHMMSS/checkpoints/v2/round_000050
```

Resume checks the source hash, split, feature schema, model hyperparameters,
workflow version, and scikit-learn version. The earlier run stays intact.

## Validate a saved model

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.valid model.path=AdaBoost/outputs/run_YYYYMMDD_HHMMSS/v1_model.pkl
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.valid model.path=AdaBoost/outputs/run_YYYYMMDD_HHMMSS/v2_model.pkl
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m AdaBoost.valid model.path=AdaBoost/outputs/run_YYYYMMDD_HHMMSS/v2_model.pkl validation.protocol=v1
```

`validation.protocol=native` uses the saved model's original validation
months and weights. Evaluating v2 with `validation.protocol=v1` scores the
same 202404–202405 rows, without month weights, as v1's native result. This
same-period comparison helps identify effects from the validation period;
the models still have different training periods. A protocol is rejected if
its validation months overlap the model's training months.

`conf/valid.yaml` controls the model, data, output, and prediction worker
count. The validator uses the original source CSV by default and verifies its
size and SHA-256 hash. An alternate `data.path` may contain just the required
validation months, but its predictor columns and order must match the saved
schema. It writes a new validation subdirectory beside the model with an HTML
report, prediction rows, overall and monthly metrics, protocol comparison,
confusion matrix, calibration and threshold tables, and PNG plots for the
training curve, monthly metrics, confusion matrix, ROC/PR, calibration, and
score distribution. Misclassification rate is the primary score; ROC AUC,
average precision, log loss, and Brier score are diagnostic.

The supplied CSV is undersampled. Its validation scores describe those rows,
not an independent test set or a population event rate.

## Verify

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m unittest discover -s AdaBoost/tests -v
```

The regression tests compare the SAMME implementation with scikit-learn on
complete data and compare uninterrupted training with checkpoint continuation.
The integration test covers both protocols, native and cross-protocol
validation, missing values, and unknown text categories. Numerical provenance
and an actual-data timing sample are in `NUMERICAL_PROVENANCE.md`.
