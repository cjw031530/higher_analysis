# Elastic Net numerical record

- Version 1.0.0 (2026-10-06): on a real-data sample, 500 epochs produced the
  same `0.175` misclassification rate as scikit-learn, with mean absolute
  probability difference `8.62e-5`.
- Resuming the full-data v2 run from epoch 180 reproduced the uninterrupted
  epoch-200 coefficients, history, and optimizer arrays exactly.
- The two-model 200-epoch run took 2:38.83 and 2,467,032 KiB peak memory.
  Both models reached the epoch cap before the configured convergence
  tolerance. The saved run is in `outputs/run_20261006_165020/`.
