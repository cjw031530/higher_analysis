# Numerical provenance: Ensemble workflow 1.0.0

The equal-probability ensemble is new. It uses all non-`TARGET` predictors,
train-only categorical vocabularies, a fixed 0.5 classification threshold,
and exact per-month validation weights. Its member objectives are binary
logistic losses. The common boosting-round budget is selected using inner
training-period holdout zero-one loss; outer validation scores are display
and report data only. The default final checkpoint block size is 20 rounds.

Versions in the synthetic verification environment were CatBoost 1.2.10,
LightGBM 4.7.0, XGBoost 3.4.1, NumPy 2.5.1, pandas 3.0.3, and Python 3.12.13.

## Numerical regression comparison

The representative synthetic comparison in
`benchmarks/synthetic_benchmark.py` uses 8,000 rows, 64 numeric columns with
4% missing entries, one categorical column, 6,000 training rows, 2,000
validation rows, 20 rounds, and two threads per member. It compares one
continuous 20-round fit against two 10-round fits using native continuation.
The maximum absolute probability difference was 0.0 for CatBoost, LightGBM,
and XGBoost on this input. Both ensemble misclassification rates were 0.0715.
This comparison does not guarantee exact equivalence on the full CSV. Native
continuation is part of this workflow's numerical method, and checkpoint
interval and library versions are recorded for reproducibility.

The synthetic pipeline test also compared a four-round uninterrupted workflow
with a workflow resumed after round two. Their outer validation error curves
agreed within 1e-12. Changing *only* the outer v2 validation labels left the
inner selected rounds, inner curve, and all saved member probabilities
unchanged; only the display-only outer score changed.

## Runtime and memory benchmark

On the same 8,000-row synthetic workload, continuous fitting took 0.3451 s,
two-block fitting took 0.2808 s, and peak process RSS was 384.92 MiB. These
short timings are noisy and should not be extrapolated to the 162,355-row,
560-column source. No full-source training or validation was run here.

Run the benchmark again with the designated Python:

```bash
'/home/cjw031530/1/rhythm rl/baseline/.venv/bin/python3' -m Ensemble.benchmarks.synthetic_benchmark
```
