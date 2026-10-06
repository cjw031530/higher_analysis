# Project State

The project predicts the binary `TARGET` in
`kcb_202306_202405_undersampled_1to4.csv` from every other column, including
`LNMON`. The root CSV is read-only. The EDA notebook describes the data.

The implemented model family is XGBoost, with two chronological training and
validation variants. Training, checkpoint continuation, and saved-model
validation are implemented in `XGBoost/`. Configuration lives in
`XGBoost/conf/`; dependencies live in `XGBoost/requirements.txt`.

The primary validation score is misclassification rate at a 0.5 probability
cutoff. Version 2 applies 60%/30%/10% weights to its validation months.
Small-data and checkpoint tests have passed, but the full CSV has not been
trained in this workspace. There is no independent test set.
