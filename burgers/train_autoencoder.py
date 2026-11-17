"""Train the Burgers autoencoder (64 -> 16-dim latent) on normalized frames.

Writes encoder.pkl, decoder.pkl and norm_stats.npy (per-grid-point mean/std
pooled over trajectories and time) to --out-dir.

    python -m burgers.train_autoencoder --data data/burgers/StochBurgersTrain.npy \
        --out-dir runs/burgers/autoencoder
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

from burgers.autoencoder import Encoder1D, Decoder1D


@nnx.jit
def train_step(enc, dec, optimizer, x):
    def loss_fn(enc, dec):
        x_hat = jax.vmap(dec)(jax.vmap(enc)(x))
        return jnp.mean(jnp.sum((x_hat - x) ** 2, axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1))(enc, dec)
    optimizer.update((enc, dec), grads)
    return loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--enc-ch', type=int, default=16)
    p.add_argument('--lat-ch', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--n-steps', type=int, default=50_000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)

    raw = np.load(a.data)                                   # [N, T, 64]
    mean = raw.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = raw.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    frames = ((raw - mean) / std).reshape(-1, raw.shape[-1]).astype(np.float32)
    np.save(os.path.join(a.out_dir, 'norm_stats.npy'), np.stack([mean, std]))

    rngs = nnx.Rngs(a.seed)
    enc = Encoder1D(a.seq_len, a.enc_ch, a.lat_ch, rngs=rngs)
    dec = Decoder1D(a.seq_len, a.enc_ch, a.lat_ch, rngs=rngs)
    optimizer = nnx.Optimizer((enc, dec), optax.adamw(optax.cosine_decay_schedule(a.lr, a.n_steps)),
                              wrt=nnx.Param)

    csv_file = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'elapsed_s'])
    key = jax.random.PRNGKey(a.seed)
    t0 = time.time()
    for step in range(a.n_steps):
        key, k = jax.random.split(key)
        idx = jax.random.randint(k, (a.batch_size,), 0, frames.shape[0])
        loss = train_step(enc, dec, optimizer, jnp.array(frames[np.array(idx)]))
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(loss):.4e}  elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(loss), round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    for name, net in (('encoder', enc), ('decoder', dec)):
        with open(os.path.join(a.out_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(jax.tree.map(np.array, nnx.state(net, nnx.Param)), f)
    print(f'saved encoder/decoder -> {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
