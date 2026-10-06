# Project State

## Current status

The repository contains the read-only KCB monthly CSV and its completed EDA
notebook. The first predictive baseline is implemented in `XGBoost/`. It defines
two chronological training and validation configurations and writes models,
probabilities, metrics, and version information to a new ignored run directory.

No independent test data is available. Full-data training has not been run as
part of the initial implementation. A 384-row, 12-month smoke test completed
both models using eight trees each. It verified validation row counts, exact
60/30/10 month weight sums, model reload and prediction agreement, and unseen
validation category handling. Runtime dependencies were installed in the
workspace Python environment; the declared dependencies are in
`XGBoost/requirements.txt`.

## Data and EDA findings

The CSV contains 162,355 rows from 202306 through 202405, 560 columns total,
and 558 non-time feature columns. Four of those features are text categories.
The original data and EDA export CSV files at the project root are not edited
by the model workflow.
