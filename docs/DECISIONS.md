# Decisions

- Use chronological validation: v1 trains through 202403 and validates on
  202404–202405; v2 trains through 202402 and validates on 202403–202405.
  Normalize v2 row weights so those three months contribute 60%, 30%, and 10%
  to validation metrics.
- Keep `LNMON` and every other non-target column as input. Learn categorical
  vocabularies from training rows only, and preserve the root CSV unchanged.
- Keep `binary:logistic` for fitting because raw misclassification loss is not
  differentiable. Use XGBoost `error` for the training curve and report
  misclassification rate at probability > 0.5 as the final validation score.
- Store configuration in Hydra YAML, dependencies in
  `XGBoost/requirements.txt`, portable final models as JSON, and resumable
  same-version checkpoints separately. The current error-history workflow is
  version 3.1.0; earlier log-loss checkpoints cannot resume under it, but
  their final models can still be validated.
