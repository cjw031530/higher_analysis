# AdaBoost numerical record

- Version 1.0.0 (2026-10-06): a reference comparison agreed with scikit-learn
  within `1e-12`; checkpoint continuation matched an uninterrupted fit. The
  three-round actual-data timing sample used 1,612 MiB peak memory.
- Version 1.1.0 (2026-10-06): sequential and concurrent fits produced the
  same initial results. The actual-data timing sample took 11.154 seconds
  sequentially and 5.954 seconds concurrently.
