#!/bin/bash
# Rayleigh-Benard convection: data, training, Table 2, and figure.
# Usage: bash scripts/run_rb.sh   (from the repository root; one GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}/rb
RUNS=${RUNS:-runs}/rb
FIGS=${FIGS:-figures}/rb
PY=${PYTHON:-python}

# 20000 trajectories per mu in {13.5, ..., 14.2} (train) and mu = 13.65 (test).
# Shipped in data/rb; regenerated only if missing.
[ -f $DATA/RB_test_mu13.65.npy ] || $PY -m rb.generate_data --out-dir $DATA

$PY -m rb.train --data-dir $DATA --out-dir $RUNS/model

# Table 2: Sinkhorn divergence at every time step and the (x6, x7) circulation QoI.
$PY -m rb.evaluate --run-dir $RUNS/model --test-data $DATA/RB_test_mu13.65.npy \
    --out-json $RUNS/eval/metrics_mu13.65.json

$PY -m rb.plot_histograms --run-dir $RUNS/model --test-data $DATA/RB_test_mu13.65.npy --out-dir $FIGS
