"""Stochastic 9-mode Rayleigh-Benard convection data (Reiterer et al. 1998).

    dX = f_mu(X) dt + sigma dW,  X in R^9,

with the square-planform vector field of Reiterer et al. (Prandtl 0.5,
wavenumber a = 1/2), sigma = 0.05, Euler-Maruyama with dt = 0.01 on [0, 20]
and initial conditions X(0) ~ N(0, 0.02^2 I). Writes one file per control
parameter mu: RB_train_mu{mu}.npy for mu in 13.5..14.2 and RB_test_mu13.65.npy,
each [n_particles, n_steps+1, 9].
"""

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

_A = 0.5
_B1 = 4 * (1 + _A ** 2) / (1 + 2 * _A ** 2)
_B2 = (1 + 2 * _A ** 2) / (2 * (1 + _A ** 2))
_B3 = 2 * (1 - _A ** 2) / (1 + _A ** 2)
_B4 = _A ** 2 / (1 + _A ** 2)
_B5 = 8 * _A ** 2 / (1 + 2 * _A ** 2)
_B6 = 4 / (1 + 2 * _A ** 2)
SIGMA_PRANDTL = 0.5

MU_TRAIN = [13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2]
MU_TEST = [13.65]


def f_rb(C, r):
    """Reiterer et al. (1998) eq. 18 for a single state C in R^9."""
    s = SIGMA_PRANDTL
    C1, C2, C3, C4, C5, C6, C7, C8, C9 = C
    dC1 = -s * _B1 * C1 - C2 * C4 + _B4 * C4 ** 2 + _B3 * C3 * C5 - s * _B2 * C7
    dC2 = -s * C2 + C1 * C4 - C2 * C5 + C4 * C5 - s * C9 / 2
    dC3 = -s * _B1 * C3 + C2 * C4 - _B4 * C2 ** 2 - _B3 * C1 * C5 + s * _B2 * C8
    dC4 = -s * C4 - C2 * C3 - C2 * C5 + C4 * C5 + s * C9 / 2
    dC5 = -s * _B5 * C5 + C2 ** 2 / 2 - C4 ** 2 / 2
    dC6 = -_B6 * C6 + C2 * C9 - C4 * C9
    dC7 = -_B1 * C7 - r * C1 + 2 * C5 * C8 - C4 * C9
    dC8 = -_B1 * C8 + r * C3 - 2 * C5 * C7 + C2 * C9
    dC9 = -C9 - r * C2 + r * C4 - 2 * C2 * C6 + 2 * C4 * C6 + C4 * C7 - C2 * C8
    return jnp.array([dC1, dC2, dC3, dC4, dC5, dC6, dC7, dC8, dC9])


_f_rb_batch = jax.jit(jax.vmap(f_rb, in_axes=(0, None)))


def generate_trajectories(r, n_particles, n_steps, dt, sigma_noise, base_seed, chunk_size):
    """Euler-Maruyama trajectories -> float32 [n_particles, n_steps+1, 9]."""
    r_f, dt_f, sig_f = jnp.float32(r), jnp.float32(dt), jnp.float32(sigma_noise)
    sqrt_dt_f = jnp.float32(np.sqrt(dt))

    def scan_fn(x, noise_t):
        x_next = x + _f_rb_batch(x, r_f) * dt_f + sig_f * sqrt_dt_f * noise_t
        return x_next, x_next

    chunks = []
    for start in range(0, n_particles, chunk_size):
        n_chunk = min(chunk_size, n_particles - start)
        key = jax.random.PRNGKey(base_seed + start)
        key, k0, kn = jax.random.split(key, 3)
        x0 = jax.random.normal(k0, (n_chunk, 9), dtype=jnp.float32) * 0.02
        noise = jax.random.normal(kn, (n_steps, n_chunk, 9), dtype=jnp.float32)
        _, xs = jax.lax.scan(scan_fn, x0, noise)
        traj = jnp.concatenate([x0[None], xs], axis=0).transpose(1, 0, 2)
        chunks.append(np.array(traj, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-particles', type=int, default=20_000)
    p.add_argument('--n-steps', type=int, default=2000)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sigma-noise', type=float, default=0.05)
    p.add_argument('--chunk-size', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mu', type=float, nargs='+', default=None,
                   help='subset of control parameters (default: all train + test)')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)
    for mu in (a.mu if a.mu is not None else MU_TRAIN + MU_TEST):
        tag = 'test' if (mu in MU_TEST and mu not in MU_TRAIN) else 'train'
        path = os.path.join(a.out_dir, f'RB_{tag}_mu{mu:.2f}.npy')
        t0 = time.time()
        data = generate_trajectories(mu, a.n_particles, a.n_steps, a.dt, a.sigma_noise,
                                     a.seed + int(round(mu * 100)), a.chunk_size)
        np.save(path, data)
        print(f'mu={mu:.2f}  {data.shape}  {time.time() - t0:.0f}s -> {path}', flush=True)


if __name__ == '__main__':
    main()
