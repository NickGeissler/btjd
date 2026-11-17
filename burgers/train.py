"""Train the BTJD Burgers generator in the latent space of a frozen autoencoder.

The generator is conditioned on h_t + sigma * N(0, I), sigma = --noise-frac
times the std of the training latents, while the joint samples entering the
Sinkhorn field carry the clean h_t. Defaults are the configuration reported in
the paper: eps=0.1, 20 Sinkhorn iterations, h=0.1, 4 time indices x 1024
particles per step, 100k steps, noise-frac 0.01, no residual connection.

    python -m burgers.train --data data/burgers/StochBurgersTrain.npy \
        --enc-dir runs/burgers/autoencoder --out-dir runs/burgers/generator
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
from burgers.model import LatentBTNet, load_autoencoder


@partial(nnx.jit, static_argnums=(7, 10))
def train_step(model, optimizer, batch_pairs, x_t_clean, x_t_cond, z, t, times_per_step,
               h, eps, iters, w_xt):
    """batch_pairs: [n_t, n_B, 2d]; x_t_clean: [n_t, n_B, d];
    x_t_cond, z: [2*n_t, n_B, d] (perturbed conditioning for both model batches)."""
    def loss_fn(model):
        gen = jax.vmap(jax.vmap(model, in_axes=(0, 0, None)), in_axes=(0, 0, 0))(x_t_cond, z, t)
        gen_b1 = gen[:times_per_step]
        gen_b2 = gen[times_per_step:]
        d = gen_b1.shape[-1] // 2
        joint_b1 = jnp.concatenate([x_t_clean, gen_b1[..., d:]], axis=-1)
        joint_b2 = jnp.concatenate([x_t_clean, gen_b2[..., d:]], axis=-1)
        proj = jax.vmap(sinkhorn_projection, in_axes=(0, 0, None, None, None))
        T_pos = proj(jax.lax.stop_gradient(joint_b1), batch_pairs, eps, iters, w_xt)
        T_neg = proj(jax.lax.stop_gradient(joint_b1), jax.lax.stop_gradient(joint_b2),
                     eps, iters, w_xt)
        V = T_pos - T_neg
        target = jax.lax.stop_gradient(gen_b1[..., d:]) + h * V[..., d:]
        return jnp.mean(jnp.sum((gen_b1[..., d:] - target) ** 2, axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def encode_dataset(enc, raw, mean, std, batch_size=2048):
    """[N, T, 64] -> flattened latents [N, T, lat_dim]."""
    n, n_t, L = raw.shape
    frames = ((raw - mean) / std).astype(np.float32).reshape(n * n_t, L)
    enc_jit = jax.jit(jax.vmap(enc))
    out = []
    for i in range(0, len(frames), batch_size):
        batch = jnp.array(frames[i:i + batch_size])
        out.append(np.array(enc_jit(batch).reshape(len(batch), -1)))
    return np.concatenate(out, axis=0).reshape(n, n_t, -1)


def train(model, pairs, lat_dim, h, eps, iters, w_xt, n_steps, times_per_step, part_per_step,
          noise_frac, lr, key, out_dir):
    optimizer = nnx.Optimizer(model, optax.adamw(optax.cosine_decay_schedule(lr, n_steps)),
                              wrt=nnx.Param)
    n_times, n_traj, _ = pairs.shape
    sigma = noise_frac * float(np.std(pairs[:, :, :lat_dim]))
    print(f'conditioning noise sigma = {sigma:.4f}', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    csv_file = open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'elapsed_s'])
    t0 = time.time()
    for step in range(n_steps):
        key, k_t, k_p, k_z, _, k_n, _ = jax.random.split(key, 7)
        t_idx = np.array(jax.random.choice(k_t, n_times, shape=(times_per_step,), replace=False))
        p_idx = np.array(jax.random.choice(k_p, n_traj, shape=(part_per_step,), replace=False))
        batch_pairs = jnp.array(pairs[t_idx][:, p_idx])
        x_t = batch_pairs[:, :, :lat_dim]
        x_cond = x_t + sigma * jax.random.normal(k_n, x_t.shape)
        x_cond = jnp.tile(x_cond, (2, 1, 1))
        z = jax.random.normal(k_z, (2 * times_per_step, part_per_step, lat_dim))
        t = jnp.tile(jnp.array(t_idx), 2)
        loss = train_step(model, optimizer, batch_pairs, x_t, x_cond, z, t, times_per_step,
                          h, eps, iters, w_xt)
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(loss):.4e}  elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(loss), round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    with open(os.path.join(out_dir, 'ckpt_final.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(np.array, nnx.state(model, nnx.Param)), f)
    print(f'saved {out_dir}/ckpt_final.pkl', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--enc-dir', required=True, help='output of burgers.train_autoencoder')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--enc-ch', type=int, default=16)
    p.add_argument('--lat-ch', type=int, default=4)
    p.add_argument('--gen-ch', type=int, default=32)
    p.add_argument('--n-blocks', type=int, default=3)
    p.add_argument('--residual', action='store_true',
                   help='predict h_{t+1} = h_t + g (the paper model does not)')
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--sinkhorn-iters', type=int, default=20)
    p.add_argument('--h', type=float, default=0.1)
    p.add_argument('--w-xt', type=float, default=1.0)
    p.add_argument('--noise-frac', type=float, default=0.01)
    p.add_argument('--n-steps', type=int, default=100_000)
    p.add_argument('--times-per-step', type=int, default=4)
    p.add_argument('--part-per-step', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    enc, _, mean, std = load_autoencoder(a.enc_dir, a.seq_len, a.enc_ch, a.lat_ch)
    raw = np.load(a.data)
    latents = encode_dataset(enc, raw, mean, std)
    n_times = raw.shape[1]
    del raw
    lat_dim = latents.shape[-1]
    pairs = np.concatenate([latents[:, :-1], latents[:, 1:]], axis=-1).transpose(1, 0, 2)
    del latents
    model = LatentBTNet(a.seq_len, a.lat_ch, a.gen_ch, a.n_blocks, T_max=n_times - 1,
                        residual=a.residual, rngs=nnx.Rngs(a.seed))
    train(model, pairs, lat_dim, a.h, a.eps, a.sinkhorn_iters, a.w_xt, a.n_steps,
          a.times_per_step, a.part_per_step, a.noise_frac, a.lr, jax.random.PRNGKey(a.seed),
          a.out_dir)


if __name__ == '__main__':
    main()
