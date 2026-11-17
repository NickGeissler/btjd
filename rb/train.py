"""Train the BTJD Rayleigh-Benard model.

Defaults are the configuration reported in the paper: 3x512 MLP, sinusoidal
time embedding with 50 frequencies plus normalized mu, Sinkhorn eps=0.1 with
20 iterations, h=0.1, 2 values of mu x 4 time indices x 1024 particles per
step.

    python -m rb.train --data-dir data/rb --out-dir runs/rb
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
from rb.generate_data import MU_TRAIN
from rb.model import BTMLP, MU_MIN, MU_MAX, normalize_mu


@partial(nnx.jit, static_argnums=(6, 7, 10))
def train_step(model, optimizer, batch_pairs, x_t, z, t, n_mu, times_per_step, h, eps, iters,
               w_xt, mu):
    """batch_pairs: [n_mu, n_t, n_B, 2d]; x_t, z: [2*n_mu*n_t, n_B, d];
    t: [2*n_mu*n_t]; mu: [2*n_mu*n_t, 1]. One Sinkhorn problem per (mu, t)."""
    def loss_fn(model):
        gen = jax.vmap(jax.vmap(model, in_axes=(0, 0, None, None)),
                       in_axes=(0, 0, 0, 0))(x_t, z, t, mu)
        half = n_mu * times_per_step
        n_B = gen.shape[1]
        gen_b1 = gen[:half].reshape(n_mu, times_per_step, n_B, -1)
        gen_b2 = gen[half:].reshape(n_mu, times_per_step, n_B, -1)
        proj = jax.vmap(jax.vmap(sinkhorn_projection, in_axes=(0, 0, None, None, None)),
                        in_axes=(0, 0, None, None, None))
        T_pos = proj(jax.lax.stop_gradient(gen_b1), batch_pairs, eps, iters, w_xt)
        T_neg = proj(jax.lax.stop_gradient(gen_b1), jax.lax.stop_gradient(gen_b2), eps, iters, w_xt)
        V = T_pos - T_neg
        V = V.at[:, :, :, :gen_b1.shape[-1] // 2].set(0.0)
        target = jax.lax.stop_gradient(gen_b1) + h * V
        return jnp.mean(jnp.sum((gen_b1 - target) ** 2, axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def load_training_data(data_dir, mu_train):
    """Globally normalized (x_t, x_{t+1}) pairs per mu -> list of [T, N, 18]."""
    raw = [np.load(os.path.join(data_dir, f'RB_train_mu{mu:.2f}.npy')) for mu in mu_train]
    n_total = len(raw) * raw[0].shape[0] * raw[0].shape[1]
    mean = np.zeros(raw[0].shape[-1], dtype=np.float64)
    for r in raw:
        mean += r.sum(axis=(0, 1), dtype=np.float64) / n_total
    mean = mean.astype(np.float32)
    var = np.zeros(raw[0].shape[-1], dtype=np.float64)
    for r in raw:
        var += ((r - mean) ** 2).sum(axis=(0, 1), dtype=np.float64) / n_total
    std = np.sqrt(var).astype(np.float32)
    pairs = []
    for i, arr in enumerate(raw):
        x = (arr - mean) / std
        pairs.append(np.concatenate([x[:, :-1], x[:, 1:]], axis=-1).transpose(1, 0, 2))
        raw[i] = None
    return pairs, mean, std


def train(model, data_dir, mu_train, h, eps, iters, w_xt, n_steps, n_mu, times_per_step,
          part_per_step, lr, key, out_dir):
    optimizer = nnx.Optimizer(model, optax.adamw(optax.cosine_decay_schedule(lr, n_steps)),
                              wrt=nnx.Param)
    pairs, mean, std = load_training_data(data_dir, mu_train)
    n_times, n_traj, d2 = pairs[0].shape
    d = d2 // 2
    mu_norm = np.array([normalize_mu(mu) for mu in mu_train], dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    csv_file = open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'elapsed_s'])
    t0 = time.time()
    for step in range(n_steps):
        key, k_mu, k_times, k_part, k_z = jax.random.split(key, 5)
        mu_idx = np.array(jax.random.choice(k_mu, len(mu_train), shape=(n_mu,), replace=False))
        t_idx = np.array(jax.random.choice(k_times, n_times, shape=(n_mu * times_per_step,),
                                           replace=False)).reshape(n_mu, times_per_step)
        p_idx = np.array(jax.random.choice(k_part, n_traj, shape=(part_per_step,), replace=False))
        batch_pairs = jnp.array(np.stack([pairs[m][t_idx[i]][:, p_idx]
                                          for i, m in enumerate(mu_idx)]))
        t = jnp.tile(jnp.array(t_idx.flatten()), 2)
        mu = jnp.array(np.tile(np.repeat(mu_norm[mu_idx], times_per_step), 2)[:, None])
        x_t = jnp.tile(batch_pairs[..., :d].reshape(n_mu * times_per_step, part_per_step, d),
                       (2, 1, 1))
        z = jax.random.normal(k_z, shape=(2 * n_mu * times_per_step, part_per_step, d))
        loss = train_step(model, optimizer, batch_pairs, x_t, z, t, n_mu, times_per_step,
                          h, eps, iters, w_xt, mu)
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(loss):.4e}  elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(loss), round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    np.save(os.path.join(out_dir, 'norm_stats.npy'), np.stack([mean, std]))
    np.save(os.path.join(out_dir, 'mu_range.npy'), np.array([MU_MIN, MU_MAX], dtype=np.float32))
    state = jax.tree.map(np.array, nnx.state(model, nnx.Param))
    with open(os.path.join(out_dir, 'ckpt_final.pkl'), 'wb') as f:
        pickle.dump(state, f)
    print(f'saved {out_dir}/ckpt_final.pkl', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='directory with RB_train_mu*.npy')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--T-max', type=int, default=2000)
    p.add_argument('--n-freq', type=int, default=50)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--sinkhorn-iters', type=int, default=20)
    p.add_argument('--h', type=float, default=0.1)
    p.add_argument('--w-xt', type=float, default=1.0)
    p.add_argument('--n-steps', type=int, default=100_000)
    p.add_argument('--n-mu-per-step', type=int, default=2)
    p.add_argument('--times-per-step', type=int, default=4)
    p.add_argument('--part-per-step', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    model = BTMLP(9, [a.hidden] * a.n_layers, a.T_max, a.n_freq, rngs=nnx.Rngs(a.seed))
    train(model, a.data_dir, MU_TRAIN, a.h, a.eps, a.sinkhorn_iters, a.w_xt, a.n_steps,
          a.n_mu_per_step, a.times_per_step, a.part_per_step, a.lr,
          jax.random.PRNGKey(a.seed), a.out_dir)


if __name__ == '__main__':
    main()
