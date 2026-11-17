"""Block-triangular MLP generator for Rayleigh-Benard, conditioned on the
Rayleigh parameter mu (normalized to [0, 1] over the training range)."""

import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from duffing.model import BasicMLP

MU_MIN, MU_MAX = 13.5, 14.2


class BTMLP(nnx.Module):
    """x_{t+1} = x_t + g2(x_t, z, sinusoidal(t), mu); the first block is the identity."""

    def __init__(self, data_size, hidden_arr, T_max, n_freq=50, cond_dim=1, rngs=nnx.Rngs(0)):
        self.T_max = T_max
        self.n_freq = n_freq
        self.g2 = BasicMLP(2 * data_size + 2 * n_freq + cond_dim, data_size, hidden_arr, rngs)

    def _t_enc(self, t):
        t_f = jnp.asarray(t, dtype=jnp.float32) / float(self.T_max)
        freqs = jnp.exp(jnp.linspace(0.0, jnp.log(float(self.T_max)), self.n_freq))
        return jnp.concatenate([jnp.sin(freqs * t_f), jnp.cos(freqs * t_f)])

    def step(self, x_t, z, t, cond):
        ctx = jnp.concatenate([self._t_enc(t), cond])
        return self.g2(jnp.concatenate([x_t, z, ctx])) + x_t

    def __call__(self, x_t, z, t, cond):
        return jnp.concatenate([x_t, self.step(x_t, z, t, cond)])


def normalize_mu(mu):
    return (mu - MU_MIN) / (MU_MAX - MU_MIN)


def load_model(run_dir, hidden=512, n_layers=3, T_max=2000, n_freq=50):
    model = BTMLP(9, [hidden] * n_layers, T_max, n_freq, rngs=nnx.Rngs(0))
    with open(os.path.join(run_dir, 'ckpt_final.pkl'), 'rb') as f:
        nnx.update(model, pickle.load(f))
    norm = np.load(os.path.join(run_dir, 'norm_stats.npy'))
    return model, norm[0], norm[1]


def rollout(model, x0_norm, T, mu, key):
    """Autoregressive rollout in normalized units at parameter mu -> [N, T+1, 9]."""
    cond = jnp.array([normalize_mu(mu)], dtype=jnp.float32)
    step = jax.jit(jax.vmap(model.step, in_axes=(0, 0, None, None)))
    x = jnp.asarray(x0_norm)
    xs = [np.array(x)]
    for t in range(T):
        key, sk = jax.random.split(key)
        x = step(x, jax.random.normal(sk, x.shape), jnp.array(t), cond)
        xs.append(np.array(x))
    return np.stack(xs, axis=1)
