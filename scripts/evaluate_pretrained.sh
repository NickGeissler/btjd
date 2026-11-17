#!/bin/bash
# Reproduce all tables and figures from the released weights in checkpoints/ (no training).
# Reads the shipped data/ (see README) and needs one GPU.
# Usage: bash scripts/evaluate_pretrained.sh [duffing|rb|burgers|cfd ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
OUT=${OUT:-results_pretrained}
PY=${PYTHON:-python}
CK=checkpoints
DOMAINS=${*:-duffing rb burgers cfd}

for d in $DOMAINS; do
case $d in
duffing)
    D=$DATA/duffing
    for set in paper:w2_distIC fixed_x0:w2_x0; do
        $PY -m duffing.evaluate --run-dir $CK/duffing --test-data $D/${set%%:*}/test.npy \
            --out-json $OUT/duffing/${set##*:}.json
    done
    for set in pool_distIC:qoi_distIC pool_x0:qoi_x0; do
        $PY -m duffing.evaluate --run-dir $CK/duffing --test-data $D/${set%%:*}/test.npy --skip-w2 \
            --out-json $OUT/duffing/${set##*:}.json
    done
    $PY -m duffing.plot_ensemble --run-dir $CK/duffing --test-data $D/paper/test.npy --ic-traj 0 --seed 7 \
        --out-dir $OUT/duffing
    $PY -m duffing.plot_snapshots --run-dir $CK/duffing --test-data $D/paper/test.npy --out-dir $OUT/duffing
    ;;
rb)
    $PY -m rb.evaluate --run-dir $CK/rb --test-data $DATA/rb/RB_test_mu13.65.npy \
        --out-json $OUT/rb/metrics_mu13.65.json
    $PY -m rb.plot_histograms --run-dir $CK/rb --test-data $DATA/rb/RB_test_mu13.65.npy --out-dir $OUT/rb
    ;;
burgers)
    ARGS="--run-dir $CK/burgers/generator --enc-dir $CK/burgers/autoencoder
          --test-data $DATA/burgers/StochBurgersTest.npy"
    $PY -m burgers.evaluate $ARGS --out-json $OUT/burgers/metrics_test.json
    $PY -m burgers.plot_snapshots $ARGS --out-dir $OUT/burgers
    ;;
cfd)
    # data/cfd/test_{normalized,latent}.npy are stochasticJCFDTest.npy passed through
    # cfd.normalize_data --scale-from checkpoints/cfd/train_normalized_scale.npy and
    # cfd.encode_to_latent --ae-dir checkpoints/cfd/autoencoder.
    SCALE=$($PY -c "import numpy as np; print(repr(float(np.load('$CK/cfd/train_normalized_scale.npy'))))")
    ARGS="--ckpt $CK/cfd/generator/ckpt_final.pkl --ae-dir $CK/cfd/autoencoder
          --latent-data $DATA/cfd/test_latent.npy --pixel-data $DATA/cfd/test_normalized.npy --data-scale $SCALE"
    $PY -m cfd.evaluate $ARGS --out-json $OUT/cfd/metrics_test.json
    $PY -m cfd.plot_snapshots $ARGS --out-dir $OUT/cfd
    $PY -m cfd.plot_ensemble $ARGS --op-rollouts $DATA/cfd/operator_rollouts_ic0.npz --out-dir $OUT/cfd
    ;;
*) echo "unknown domain $d"; exit 1 ;;
esac
done
