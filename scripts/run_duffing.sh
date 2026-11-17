#!/bin/bash
# Duffing oscillator: data, training, Table 1, and figures.
# Usage: bash scripts/run_duffing.sh   (from the repository root; one GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}/duffing
RUNS=${RUNS:-runs}/duffing
FIGS=${FIGS:-figures}/duffing
PY=${PYTHON:-python}

# Data (shipped in data/duffing; regenerated only if missing): 5000 train / 5000 test
# trajectories with X(0) ~ N([0,-10], I), a fixed-IC test set, and two 200k-trajectory
# pools for the trajectory QoI.
gen() { [ -f "${@: -1}/test.npy" ] || $PY -m duffing.generate_data "$@"; }
gen --M-train 5000 --M-test 5000 --seed 0 --out-dir $DATA/paper
gen --M-train 0 --M-test 5000 --seed 7 --fixed-x0 0 -10 --out-dir $DATA/fixed_x0
gen --M-train 0 --M-test 200000 --seed 13 --noise-seed 101 --out-dir $DATA/pool_distIC
gen --M-train 0 --M-test 200000 --seed 14 --noise-seed 201 --fixed-x0 0 -10 --out-dir $DATA/pool_x0

$PY -m duffing.train --data $DATA/paper/train.npy --out-dir $RUNS/model

# Table 1: sliced W2 on the 5000-trajectory test sets, QoI on the 200k pools.
$PY -m duffing.evaluate --run-dir $RUNS/model --test-data $DATA/paper/test.npy \
    --out-json $RUNS/eval/w2_distIC.json
$PY -m duffing.evaluate --run-dir $RUNS/model --test-data $DATA/fixed_x0/test.npy \
    --out-json $RUNS/eval/w2_x0.json
$PY -m duffing.evaluate --run-dir $RUNS/model --test-data $DATA/pool_distIC/test.npy --skip-w2 \
    --out-json $RUNS/eval/qoi_distIC.json
$PY -m duffing.evaluate --run-dir $RUNS/model --test-data $DATA/pool_x0/test.npy --skip-w2 \
    --out-json $RUNS/eval/qoi_x0.json

# Figures: trajectories from one IC (test trajectory 0) and phase-space snapshots.
$PY -m duffing.plot_ensemble --run-dir $RUNS/model --test-data $DATA/paper/test.npy --ic-traj 0 \
    --seed 7 --out-dir $FIGS
$PY -m duffing.plot_snapshots --run-dir $RUNS/model --test-data $DATA/paper/test.npy --out-dir $FIGS
