"""Train the turbulence autoencoder (64x64x1 <-> 16x16x4) on normalized frames.

Loss: per-sample relative MSE + 0.1 * relative MAE. AdamW (b2=0.95), gradient
clipping at 2, linear warmup then constant learning rate, EMA of the weights.
Writes encoder.pkl, decoder.pkl and their _ema versions (the EMA weights are
the ones used downstream).

--test-data is only used for held-out loss logging; note that it also draws
from the random stream, so it must be given to reproduce the paper weights.

    python -m cfd.train_autoencoder --data data/cfd/stochasticJCFDTrain_normalized.npy \
        --test-data data/cfd/stochasticJCFDTest_normalized.npy --out-dir runs/cfd/autoencoder
"""

import argparse
import csv
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from cfd.autoencoder import ConvAutoencoder


def relative_loss(x_hat, x, mae_weight, eps):
    rel_mse = jnp.mean((x_hat - x) ** 2, axis=(1, 2, 3)) / (jnp.mean(x ** 2, axis=(1, 2, 3)) + eps)
    rel_mae = (jnp.mean(jnp.abs(x_hat - x), axis=(1, 2, 3))
               / (jnp.mean(jnp.abs(x), axis=(1, 2, 3)) + eps))
    return rel_mse + mae_weight * rel_mae


@nnx.jit
def train_step(net, optimizer, x, mae_weight, eps):
    def loss_fn(net):
        per_sample = relative_loss(jax.vmap(net)(x), x, mae_weight, eps)
        return jnp.mean(per_sample), per_sample

    (_, per_sample), grads = nnx.value_and_grad(loss_fn, has_aux=True)(net)
    optimizer.update(net, grads)
    return per_sample


@nnx.jit
def eval_step(net, x, mae_weight, eps):
    return relative_loss(jax.vmap(net)(x), x, mae_weight, eps)


def load_frames(path):
    raw = np.load(path, mmap_mode='r')
    return raw[:].reshape(-1, *raw.shape[2:]).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--test-data', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--mae-weight', type=float, default=0.10)
    p.add_argument('--loss-eps', type=float, default=1e-6)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--n-steps', type=int, default=100_000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=4000)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--eval-samples', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)
    frames = load_frames(a.data)
    test_frames = load_frames(a.test_data) if a.test_data else None

    net = ConvAutoencoder(rngs=nnx.Rngs(a.seed))
    schedule = optax.join_schedules([optax.linear_schedule(0.0, a.lr, a.warmup_steps),
                                     optax.constant_schedule(a.lr)], boundaries=[a.warmup_steps])
    tx = optax.chain(optax.clip_by_global_norm(2.0),
                     optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=a.weight_decay))
    optimizer = nnx.Optimizer(net, tx, wrt=nnx.Param)
    ema = jax.tree.map(jnp.array, nnx.state(net, nnx.Param))

    @nnx.jit
    def ema_update(ema, net, decay):
        return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p,
                            ema, nnx.state(net, nnx.Param))

    csv_file = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'test_loss', 'elapsed_s'])
    key = jax.random.PRNGKey(a.seed)
    t0 = time.time()
    for step in range(a.n_steps):
        key, k = jax.random.split(key)
        idx = np.array(jax.random.randint(k, (a.batch_size,), 0, frames.shape[0]))
        loss = train_step(net, optimizer, jnp.array(frames[idx]), a.mae_weight, a.loss_eps)
        ema = ema_update(ema, net, a.ema_decay)
        if step % a.log_every == 0:
            test_loss = ''
            if test_frames is not None and step % a.eval_every == 0:
                key, k_test = jax.random.split(key)
                t_idx = np.array(jax.random.randint(k_test, (a.eval_samples,), 0,
                                                    test_frames.shape[0]))
                test_loss = float(np.mean(eval_step(net, jnp.array(test_frames[t_idx]),
                                                    a.mae_weight, a.loss_eps)))
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(np.mean(loss)):.5e}  test={test_loss}  '
                  f'elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(np.mean(loss)), test_loss, round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    for suffix, state in (('', nnx.state(net, nnx.Param)), ('_ema', ema)):
        full = jax.tree.map(np.array, state)
        for part in ('encoder', 'decoder'):
            with open(os.path.join(a.out_dir, f'{part}{suffix}.pkl'), 'wb') as f:
                pickle.dump(full[part], f)
    print(f'saved encoder/decoder (+ _ema) -> {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
