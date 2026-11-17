"""Train the BTJD turbulence generator in the latent space of the frozen autoencoder.

Each step draws --times-per-step time indices (one per equal-width bin of the
stride-4 grid) and, independently per time, --part-per-step trajectories.
The generator is conditioned on h_t + sigma * N(0, I) (sigma = --noise-frac
times the latent std), while the joint features entering the drift loss use
the clean h_t: features of (h_t, h_{t+1}) from the frozen MAE are compared
between the model batch, the data, and an independent model batch.

Defaults are the configuration reported in the paper.

    python -m cfd.train --data data/cfd/stochasticJCFDTrain_latent.npy \
        --feature-ckpt runs/cfd/mae/encoder_ema.pkl --out-dir runs/cfd/generator
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

from cfd.drift_loss import total_drift_loss
from cfd.feature_net import MultiscaleFeatureNetLatent, all_feature_keys
from cfd.generator import BlockTriangularUNet2DLatent


@partial(nnx.jit, static_argnums=(11,))
def train_step(model, feature_net, optimizer, x_t, x_t1, z1, z2, noise, t, R, w_xt, num_iter):
    """x_t, x_t1, z1, z2, noise: [n_t, n_B, 16, 16, C]; t: [n_t]."""
    C = x_t.shape[-1]
    h_cond = jax.lax.stop_gradient(x_t + noise)

    def loss_fn(model):
        run = jax.vmap(jax.vmap(model, in_axes=(0, 0, None)), in_axes=(0, 0, 0))
        gen_next = run(h_cond, z1, t)[..., C:2 * C]
        neg_next = jax.lax.stop_gradient(run(h_cond, z2, t)[..., C:2 * C])
        extract = lambda x: jax.vmap(lambda xx: all_feature_keys(feature_net, xx))(x)

        def per_t(xt, gen, pos, neg):
            f_ht = jax.lax.stop_gradient(extract(xt))
            f_gen = extract(gen)
            f_pos = jax.lax.stop_gradient(extract(pos))
            f_neg = jax.lax.stop_gradient(extract(neg))
            joint = lambda c, n: jnp.concatenate([jax.lax.stop_gradient(c) * w_xt, n], axis=-1)
            return total_drift_loss({k: joint(f_ht[k], f_gen[k]) for k in f_gen},
                                    {k: joint(f_ht[k], f_pos[k]) for k in f_pos},
                                    {k: joint(f_ht[k], f_neg[k]) for k in f_neg},
                                    R, num_iter)

        return jnp.mean(jax.vmap(per_t)(x_t, gen_next, x_t1, neg_next))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def save_params(tree, path):
    with open(path, 'wb') as f:
        pickle.dump(jax.tree.map(np.array, tree), f)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='latent training set [N, T, 16, 16, 4]')
    p.add_argument('--feature-ckpt', required=True, help='MAE encoder_ema.pkl')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--R', type=float, default=0.07,
                   help='entropic regularization; eps = R * feature dim on distance-normalized features')
    p.add_argument('--w-xt', type=float, default=1.0)
    p.add_argument('--sinkhorn-iters', type=int, default=20)
    p.add_argument('--noise-frac', type=float, default=0.02)
    p.add_argument('--time-stride', type=int, default=4)
    p.add_argument('--times-per-step', type=int, default=64)
    p.add_argument('--part-per-step', type=int, default=96)
    p.add_argument('--n-steps', type=int, default=65_000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup-steps', type=int, default=5000)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)

    raw = np.load(a.data)                              # [N, T, 16, 16, C]
    n_traj, n_t, _, _, C = raw.shape
    times = np.arange(0, n_t - a.time_stride, a.time_stride)
    sigma = a.noise_frac * float(np.std(raw[:min(n_traj, 64)]))
    print(f'{n_traj} trajectories x {len(times)} pairs (stride {a.time_stride}); '
          f'conditioning noise sigma = {sigma:.4f}', flush=True)

    rngs = nnx.Rngs(a.seed)
    feature_net = MultiscaleFeatureNetLatent(in_channels=C, rngs=rngs)
    with open(a.feature_ckpt, 'rb') as f:
        nnx.update(feature_net.encoder, pickle.load(f))
    model = BlockTriangularUNet2DLatent(in_channels=C, rngs=rngs)

    schedule = optax.join_schedules([optax.linear_schedule(0.0, a.lr, a.warmup_steps),
                                     optax.constant_schedule(a.lr)], boundaries=[a.warmup_steps])
    optimizer = nnx.Optimizer(model, optax.adamw(schedule), wrt=nnx.Param)

    csv_file = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss_mean', 'elapsed_s'])
    key = jax.random.PRNGKey(a.seed)
    bin_edges = np.linspace(0, len(times), a.times_per_step + 1).astype(int)
    losses, t0 = [], time.time()
    for step in range(a.n_steps):
        key, k_t, _, k_z1, k_z2, k_n, _ = jax.random.split(key, 7)
        k_bins = jax.random.split(k_t, a.times_per_step)
        t_idx = times[np.array([int(jax.random.randint(k_bins[i], (), bin_edges[i], bin_edges[i + 1]))
                                for i in range(a.times_per_step)])]
        x_t, x_t1 = [], []
        for ti in t_idx:
            key, k_p = jax.random.split(key)
            p_idx = np.array(jax.random.choice(k_p, n_traj, shape=(a.part_per_step,), replace=False))
            x_t.append(raw[p_idx, ti])
            x_t1.append(raw[p_idx, ti + a.time_stride])
        x_t = jnp.array(np.stack(x_t))
        x_t1 = jnp.array(np.stack(x_t1))
        z1 = jax.random.normal(k_z1, x_t.shape)
        z2 = jax.random.normal(k_z2, x_t.shape)
        noise = sigma * jax.random.normal(k_n, x_t.shape)
        loss = train_step(model, feature_net, optimizer, x_t, x_t1, z1, z2, noise,
                          jnp.array(t_idx, dtype=jnp.float32), a.R, a.w_xt, a.sinkhorn_iters)
        losses.append(float(loss))
        if step % a.log_every == 0:
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={np.mean(losses):.4e}  elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(np.mean(losses)), round(elapsed, 1)])
            csv_file.flush()
            losses.clear()
        if step > 0 and step % a.save_every == 0:
            save_params(nnx.state(model, nnx.Param), os.path.join(a.out_dir, f'ckpt_{step}.pkl'))
    csv_file.close()
    save_params(nnx.state(model, nnx.Param), os.path.join(a.out_dir, 'ckpt_final.pkl'))
    print(f'saved {a.out_dir}/ckpt_final.pkl', flush=True)


if __name__ == '__main__':
    main()
