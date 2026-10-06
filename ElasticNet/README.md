# KCB Elastic Net logistic regression

This package fits two binary Elastic Net logistic regression models to the
read-only root CSV. All 559 non-`TARGET` columns, including `LNMON`, are model
inputs. The source has 162,355 rows and no independent test set.

## Setup

The compiled SAGA kernel is a private scikit-learn API, so the workflow pins
`scikit-learn==1.9.1`. The initial `penalty="elasticnet"` setting is accepted in
that version but is deprecated for later versions. The requirements are local
to this model family:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m pip install -r ElasticNet/requirements.txt
```

## Train

From the project root:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m ElasticNet.train
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m ElasticNet.train model.max_epochs=300 checkpoint.every=25
```

Hydra configuration is in `conf/train.yaml` and `conf/protocols/default.yaml`.
The initial model uses `solver="saga"`, `penalty="elasticnet"`, `l1_ratio=0.5`,
and `C=0.1`. Version 1 trains on 202306–202403 and validates on
202404–202405. Version 2 trains on 202306–202402 and validates on
202403–202405, with validation month shares 60%, 30%, and 10%. Each month's
share is divided by that month's validation row count. The shares affect
validation metrics only. The primary score is misclassification rate at a
fixed probability cutoff of 0.5, assigning exact ties to class 0.

The fitted objective is Elastic Net penalized logistic log loss. Standard SAGA
cannot directly optimize discontinuous 0/1 misclassification loss. The latter
is monitored after each epoch and reported by `valid.py`; it does not select
the training epoch. The final model is the last completed epoch, or the epoch
where the coefficient-change tolerance is reached. The best observed
validation epoch is informational only.

Numerical predictors are median-imputed and standardized using the training
months. Numeric columns with missing training values get an additional missing
indicator. Text predictors are one-hot encoded using training-only levels;
missing and previously unseen text get a separate indicator. Constant and
duplicate original predictors are retained. The complete preprocessing schema
is saved with each model.

The two protocols train concurrently in separate threads by default. The
compiled SAGA kernel releases Python's GIL, and BLAS threads are limited to
one to avoid oversubscription. Set `runtime.parallel_models=false` to run one
protocol at a time, or `runtime.training_threads=N` to cap protocol workers.
Two protocol fits provide at most two independent heavy SAGA tasks. `tqdm`
shows CSV loading, preprocessing, epoch progress, and current validation
misclassification. An interactive Matplotlib backend can show live curves;
otherwise the saved history is plotted by `valid.py`.

Each run creates `ElasticNet/outputs/run_.../` and three final files per
protocol:

- `v1_model.pkl` / `v2_model.pkl`: final coefficients and preprocessing schema.
- `v1_hyperparameters.json` / `v2_hyperparameters.json`: configuration,
  source hash, split, package versions, and fit status.
- `v1_history.csv` / `v2_history.csv`: epoch, validation misclassification,
  and relative coefficient change.

Separate `checkpoints/v1/epoch_000020/` and corresponding v2 directories
contain `optimizer_state.npz`, `history.csv`, and `state.json`. The snapshot
includes coefficients, intercept, the SAGA gradient table, accumulated
gradients, seen-sample flags, seen count, completed epoch, step size, and the
deterministic sampling schedule. The kernel flushes pending coefficient
updates at each epoch boundary. Each epoch uses a seed derived from the fixed
base seed and epoch number, so a resumed run follows the same stochastic
samples as an uninterrupted run of this workflow. The checkpoint is published
atomically and checksummed. Resume verifies its data hash, schema,
hyperparameters, and exact NumPy, SciPy, and scikit-learn versions. This is
full optimizer-state continuation; `LogisticRegression.warm_start` is never
used. The epoch sampling schedule differs from one monolithic scikit-learn
`LogisticRegression.fit`, although the fitted objective and SAGA update kernel
are the same. Numerical comparison is recorded in `NUMERICAL_PROVENANCE.md`.

Resume one protocol into a new run directory:

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m ElasticNet.train checkpoint.resume_from=ElasticNet/outputs/run_YYYYMMDD_HHMMSS/checkpoints/v2/epoch_000020
```

The saved final pickle model should be loaded only from a trusted run.

## Validate

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m ElasticNet.valid model.path=ElasticNet/outputs/run_YYYYMMDD_HHMMSS/v1_model.pkl
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m ElasticNet.valid model.path=ElasticNet/outputs/run_YYYYMMDD_HHMMSS/v2_model.pkl validation.protocol=v1
```

`validation.protocol=native` uses the model's original validation months and
weights. `validation.protocol=v1` scores a v2 model on the same unweighted
202404–202405 period as v1's native result. The validator rejects a protocol
whose validation months overlap the model's training months. A change in the
original data source is rejected using size and SHA-256. An alternate
`data.path` can contain only the required validation months if its predictor
schema matches.

`valid.py` writes an HTML report, predictions, overall and monthly metric
tables, a same-model protocol comparison, calibration deciles, threshold
diagnostics, a confusion matrix, and PNG figures for training error, monthly
metrics, ROC/precision-recall, calibration, score distributions, and the
confusion matrix. The reported primary score is ordinary row misclassification
for v1 and month-weighted misclassification for v2. ROC AUC, average
precision, log loss, and Brier score are supplementary. Other thresholds are
diagnostics only; the final cutoff remains 0.5. These scores describe the
undersampled CSV and are not population event rates or independent test
scores.

## Verify

```bash
/home/cjw031530/1/rhythm\ rl/baseline/.venv/bin/python3 -m unittest discover -s ElasticNet/tests -v
```

The tests compare checkpoint continuation with uninterrupted epoch-scheduled
SAGA, compare the converged probabilities with scikit-learn's public
`LogisticRegression`, and exercise both protocols and cross-protocol
validation through the CLIs.
