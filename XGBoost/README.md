# XGBoost

This directory contains the XGBoost workflow for the binary `TARGET` task.
The root CSV is read-only. The shared data splits and primary metric are in
`../docs/ARCHITECTURE.md`.

Configuration is in `conf/`, dependencies are in `requirements.txt`, and run
artifacts are in `outputs/`. From the project root, use the workspace virtual
environment to run `-m XGBoost.train` or `-m XGBoost.valid model.path=PATH`.
