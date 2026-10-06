# GaussianNB numerical record

- Version 1.0.0 (2026-10-06): initial GaussianNB implementation. Numeric
  predictors use train-only median imputation and standardization. Text
  predictors use train-only one-hot levels. `var_smoothing` is chosen by
  unweighted 0/1 error on the last training month and the final model uses
  chronological `partial_fit` batches. Validation month shares affect only
  validation metrics. The final threshold is 0.5.
- The synthetic checkpoint replay test compares uninterrupted and resumed
  model means, variances, and probabilities with exact equality. Synthetic
  validation probabilities from the disk-backed matrix also match direct
  batch transformation exactly. Synthetic validation tests cover native v2
  and the same saved v2 model on v1 months.
- A prediction-path benchmark with 12,000 rows, 560 encoded features, and five
  validation updates took 0.680 seconds with repeated direct transformation
  and 0.192 seconds with one disk-backed transformation followed by repeated
  prediction. The maximum probability difference was zero. The disk-backed
  matrix occupied about 51 MiB. This synthetic timing is not a full-data
  training benchmark; full-data training is reserved for the user.
- `partial_fit` applies variance smoothing at each batch. Batch size and row
  order are therefore part of the saved model's numerical provenance and
  checkpoint compatibility checks.
- Version 1.1.0 (2026-10-06): the default training batch decreased from
  10,000 to 2,000 rows, increasing the current CSV's expected updates from
  14 to 67 for v1 and from 13 to 61 for v2. The default checkpoint interval
  changed from two to five batches, or roughly 10,000 processed rows. The
  model algorithm and saved-artifact format are unchanged.
- Regression comparison on 12,345 synthetic training rows, 3,000 validation
  rows, and 560 features used the same row order and `var_smoothing=1e-7`.
  Final 0/1 error was 0.158000 for both batch sizes. The largest absolute
  differences were `9.99e-16` for class means, `4.51e-09` for variances,
  and `3.12e-09` for predicted probabilities. In that timing sample,
  training plus validation after every update took 0.115 seconds with
  10,000-row batches and 0.118 seconds with 2,000-row batches. Full-data
  runtime may rise more because validation is scored at every update.
