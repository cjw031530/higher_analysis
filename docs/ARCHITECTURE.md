# Architecture

## Inputs and workflow

`XGBoost.train` reads the root CSV without modifying it. It validates the
monthly and binary target columns, then uses every column except `TARGET` as an
input. `LNMON` stays in the input matrix as requested. The four text columns
identified by the EDA are inferred from data types and converted to pandas
categories using training-only vocabularies; unseen validation values become
missing values. XGBoost handles numerical and categorical missing values.

The module trains two `XGBClassifier` instances. Version 1 uses 202306–202403
for training and 202404–202405 for validation. Version 2 uses 202306–202402 for
training and 202403–202405 for validation. Version 2's validation row weight is
the desired month share divided by its row count, yielding exact total month
shares of 0.6, 0.3, and 0.1. No validation weight affects fitting.

The two models run in a thread pool. Each XGBoost instance uses compiled
histogram training with a configurable number of CPU threads, half the detected
CPU count by default. A tqdm callback reports boosting iterations; another
bar reports rows read. Sequential mode is available to lower peak memory.

## Outputs

Each invocation creates a unique directory under `XGBoost/outputs/`. JSON
models retain categorical split information. Prediction CSV files include the
source row index, month, target, probability, and validation weight. Metric
JSON files include overall and monthly ROC AUC, average precision, and log loss;
version 2 also includes weighted overall metrics. `run.json` records versions,
schema, data path and size, and runtime settings.

Outputs are ignored by Git. There is no independent test evaluation, early
stopping, calibration, or threshold selection in this first baseline.
