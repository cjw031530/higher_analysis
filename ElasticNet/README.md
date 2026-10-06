# Elastic Net logistic regression

This directory contains the Elastic Net workflow for the binary `TARGET`
task. The root CSV is read-only. The shared data splits and primary metric
are in `../docs/ARCHITECTURE.md`.

Configuration is in `conf/`, dependencies are in `requirements.txt`, and run
artifacts are in `outputs/`. From the project root, use the workspace virtual
environment to run `-m ElasticNet.train` or
`-m ElasticNet.valid model.path=PATH`. Tests are in `tests/`; numerical checks
are summarized in `NUMERICAL_PROVENANCE.md`.
