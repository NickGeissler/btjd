#!/bin/bash
# Stochastic Burgers: autoencoder, generator, Table 3, and figure.
# Expects $DATA/StochBurgersTrain.npy and $DATA/StochBurgersTest.npy ([4096, 800, 64]).
# Usage: bash scripts/run_burgers.sh   (from the repository root; one GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}/burgers
RUNS=${RUNS:-runs}/burgers
FIGS=${FIGS:-figures}/burgers
PY=${PYTHON:-python}

$PY -m burgers.train_autoencoder --data $DATA/StochBurgersTrain.npy --out-dir $RUNS/autoencoder
$PY -m burgers.train --data $DATA/StochBurgersTrain.npy --enc-dir $RUNS/autoencoder \
    --out-dir $RUNS/generator

# Table 3: energy / enstrophy relative errors on all 4096 test trajectories.
$PY -m burgers.evaluate --run-dir $RUNS/generator --enc-dir $RUNS/autoencoder \
    --test-data $DATA/StochBurgersTest.npy --out-json $RUNS/eval/metrics_test.json

$PY -m burgers.plot_snapshots --run-dir $RUNS/generator --enc-dir $RUNS/autoencoder \
    --test-data $DATA/StochBurgersTest.npy --out-dir $FIGS
