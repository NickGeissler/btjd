"""Train the BTJD Duffing model (block-triangular joint drifting).

Defaults are the configuration reported in the paper: 2x512 MLP, scalar time
input, Sinkhorn eps=0.01 with 50 iterations, h=0.1, 4 time indices x 512
particles per step, 100k steps.

    python -m duffing.train --data data/duffing/paper/train.npy --out-dir runs/duffing
"""

import argparse
import csv
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from common.sinkhorn import sinkhorn_projection
from duffing.model import BTMLP


@partial(nnx.jit, static_argnums=(6, 9))
def train_step(model, optimizer, batch_pairs, x_t, z, t, times_per_step, h, eps, iters, w_xt):
    """One step of the projected joint drifting loss.

    batch_pairs: [n_t, n_B, 2d] data pairs; x_t, z: [2*n_t, n_B, d] (the two
    halves are independent model batches); t: [2*n_t].
    """
    def loss_fn(model):
        gen = jax.vmap(jax.vmap(model, in_axes=(0, 0, None)), in_axes=(0, 0, 0))(x_t, z, t)
        gen_b1 = gen[:times_per_step]
        gen_b2 = gen[times_per_step:]
        proj = jax.vmap(sinkhorn_projection, in_axes=(0, 0, None, None, None))
        T_pos = proj(jax.lax.stop_gradient(gen_b1), batch_pairs, eps, iters, w_xt)
        T_neg = proj(jax.lax.stop_gradient(gen_b1), jax.lax.stop_gradient(gen_b2), eps, iters, w_xt)
        V = T_pos - T_neg
        V = V.at[:, :, :gen_b1.shape[-1] // 2].set(0.0)   # projection onto the second block
        target = jax.lax.stop_gradient(gen_b1) + h * V
        return jnp.mean(jnp.sum((gen_b1 - target) ** 2, axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(model, data_path, h, eps, iters, w_xt, n_steps, times_per_step, part_per_step,
          lr, key, out_dir):
    optimizer = nnx.Optimizer(model, optax.adamw(optax.cosine_decay_schedule(lr, n_steps)),
                              wrt=nnx.Param)

    data = jnp.load(data_path)                       # [N, T+1, d]
    n_traj, n_times, d = data.shape
    norm_mean = jnp.mean(data, axis=(0, 1))
    norm_std = jnp.std(data, axis=(0, 1))
    data = (data - norm_mean) / norm_std
    pairs = jnp.concatenate([data[:, :-1], data[:, 1:]], axis=-1).transpose(1, 0, 2)

    os.makedirs(out_dir, exist_ok=True)
    csv_file = open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'elapsed_s'])
    t0 = time.time()
    for step in range(n_steps):
        key, k_times, k_part, k_z = jax.random.split(key, 4)
        t_idx = jax.random.choice(k_times, n_times - 1, shape=(times_per_step,), replace=False)
        p_idx = jax.random.choice(k_part, n_traj, shape=(part_per_step,), replace=False)
        batch_pairs = pairs[t_idx][:, p_idx]
        x_t = jnp.tile(batch_pairs[:, :, :d], (2, 1, 1))
        z = jax.random.normal(k_z, shape=(2 * times_per_step, part_per_step, d))
        t = jnp.tile(t_idx, 2)
        loss = train_step(model, optimizer, batch_pairs, x_t, z, t, times_per_step,
                          h, eps, iters, w_xt)
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(loss):.4e}  elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(loss), round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    np.save(os.path.join(out_dir, 'norm_stats.npy'),
            np.stack([np.array(norm_mean), np.array(norm_std)]))
    state = jax.tree.map(np.array, nnx.state(model, nnx.Param))
    with open(os.path.join(out_dir, 'ckpt_final.pkl'), 'wb') as f:
        pickle.dump(state, f)
    print(f'saved {out_dir}/ckpt_final.pkl', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='train.npy from duffing.generate_data')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--time-enc', default='scalar', choices=['scalar', 'sinusoidal'])
    p.add_argument('--T-max', type=int, default=1200, help='time normalization T')
    p.add_argument('--eps', type=float, default=0.01)
    p.add_argument('--sinkhorn-iters', type=int, default=50)
    p.add_argument('--h', type=float, default=0.1)
    p.add_argument('--w-xt', type=float, default=1.0)
    p.add_argument('--n-steps', type=int, default=100_000)
    p.add_argument('--times-per-step', type=int, default=4)
    p.add_argument('--part-per-step', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    model = BTMLP(2, [a.hidden] * a.n_layers, a.T_max, time_enc=a.time_enc,
                  rngs=nnx.Rngs(a.seed))
    train(model, a.data, a.h, a.eps, a.sinkhorn_iters, a.w_xt, a.n_steps,
          a.times_per_step, a.part_per_step, a.lr, jax.random.PRNGKey(a.seed), a.out_dir)


if __name__ == '__main__':
    main()
