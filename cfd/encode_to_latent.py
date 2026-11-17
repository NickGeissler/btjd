"""Encode a normalized vorticity dataset [N, T, 64, 64, 1] into [N, T, 16, 16, 4]
latents with the frozen (EMA) autoencoder encoder."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from cfd.autoencoder import load_autoencoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--ae-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=1024)
    a = p.parse_args()

    raw = np.load(a.data, mmap_mode='r')
    n, n_t = raw.shape[:2]
    flat = raw.reshape(n * n_t, *raw.shape[2:])
    encode = jax.jit(jax.vmap(load_autoencoder(a.ae_dir).encode))
    out = [np.array(encode(jnp.array(np.array(flat[i:i + a.batch_size], dtype=np.float32))))
           for i in range(0, n * n_t, a.batch_size)]
    latents = np.concatenate(out).reshape(n, n_t, *out[0].shape[1:])
    np.save(a.out, latents)
    print(f'{latents.shape} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
