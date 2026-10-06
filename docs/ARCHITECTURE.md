# Architecture

## Data and validation protocols

`XGBoost.train` reads the root CSV without modifying it. Every column except
`TARGET` is a predictor. Text predictors use categories learned from training
rows; unseen validation categories become missing values.

| Model | Training months | Validation months | Validation weights |
| --- | --- | --- | --- |
| v1 | 202306–202403 | 202404–202405 | Equal per row |
| v2 | 202306–202402 | 202403–202405 | 60% / 30% / 10% by month |

The month weights affect validation scores, not model fitting. No independent
test period is available.

## XGBoost workflow

Hydra YAML in `XGBoost/conf/` defines paths, hyperparameters, runtime settings,
and validation protocols. `XGBoost.train` can fit both variants concurrently,
shows tqdm progress, records validation error after every tree, and writes
periodic resumable checkpoints. Final JSON models, settings, and error histories
are stored under `XGBoost/outputs/`.

The current model uses histogram trees, native categorical support, and 3,000
trees. Its fit objective is `binary:logistic`; `eval_metric=error` monitors
misclassification at probability > 0.5. Monitoring error does not change the
fitted trees because there is no early stopping.

`XGBoost.valid` loads a final model and produces prediction rows, metric tables,
plots, and an HTML report. Its primary score is misclassification rate; v2's
native score uses the month weights above. It can also evaluate a v2 model on
v1's unweighted validation months for a same-period comparison. Older final
models with log-loss histories remain validatable.
