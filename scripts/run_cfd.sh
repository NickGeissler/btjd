#!/bin/bash
# 2D stochastically forced turbulence: autoencoder, latent encoding, MAE feature
# network, generator, Table 4, and figures.
# Inputs (data/cfd): raw vorticity stochasticJCFDTrain.npy ([2048, 1000, 64, 64, 1]) and
# stochasticJCFDTest.npy ([1024, 1000, 64, 64, 1], independent ICs, same forcing), and
# operator_rollouts_ic0.npz (operator-learning baseline for the ensemble figure).
# Derived datasets are written to $RUNS/data, not to $DATA.
# Usage: bash scripts/run_cfd.sh   (from the repository root; one GPU, ~80 GB host RAM)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}/cfd
RUNS=${RUNS:-runs}/cfd
FIGS=${FIGS:-figures}/cfd
PY=${PYTHON:-python}
DER=$RUNS/data
mkdir -p $DER

# Global normalization (the test set reuses the training constant).
$PY -m cfd.normalize_data --data $DATA/stochasticJCFDTrain.npy --out $DER/train_normalized.npy
$PY -m cfd.normalize_data --data $DATA/stochasticJCFDTest.npy --out $DER/test_normalized.npy \
    --scale-from $DER/train_normalized_scale.npy
SCALE=$($PY -c "import numpy as np; print(repr(float(np.load('$DER/train_normalized_scale.npy'))))")

# Autoencoder and latent datasets.
$PY -m cfd.train_autoencoder --data $DER/train_normalized.npy --test-data $DER/test_normalized.npy \
    --out-dir $RUNS/autoencoder
$PY -m cfd.encode_to_latent --data $DER/train_normalized.npy --ae-dir $RUNS/autoencoder \
    --out $DER/train_latent.npy
$PY -m cfd.encode_to_latent --data $DER/test_normalized.npy --ae-dir $RUNS/autoencoder \
    --out $DER/test_latent.npy

# MAE feature network and generator.
$PY -m cfd.pretrain_feature_net --data $DER/train_latent.npy --test-data $DER/test_latent.npy \
    --out-dir $RUNS/mae
$PY -m cfd.train --data $DER/train_latent.npy --feature-ckpt $RUNS/mae/encoder_ema.pkl \
    --out-dir $RUNS/generator

# Table 4: energy / enstrophy relative errors over the rollout (249 steps), all test trajectories.
ARGS="--ckpt $RUNS/generator/ckpt_final.pkl --ae-dir $RUNS/autoencoder
    --latent-data $DER/test_latent.npy --pixel-data $DER/test_normalized.npy --data-scale $SCALE"
$PY -m cfd.evaluate $ARGS --out-json $RUNS/eval/metrics_test.json

# Figures.
$PY -m cfd.plot_snapshots $ARGS --out-dir $FIGS
$PY -m cfd.plot_ensemble $ARGS --op-rollouts $DATA/operator_rollouts_ic0.npz --out-dir $FIGS
