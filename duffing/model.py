"""Block-triangular MLP generator f(x, z, t) = (x, g(x, z, t)) for Duffing."""

import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


class BasicMLP(nnx.Module):
    """MLP with SiLU activations between hidden layers."""

    def __init__(self, input_dim, output_dim, hidden_arr, rngs):
        dims = [input_dim] + list(hidden_arr) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))
            if i < len(dims) - 2:
                layers.append(nnx.silu)
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


class BTMLP(nnx.Module):
    """Block-triangular map with identity first block and residual second block.

    g2 maps (x_t, z, t_enc) to an increment; x_{t+1} = x_t + g2(...).
    time_enc='scalar' feeds t / T_max; 'sinusoidal' feeds 2*n_freq log-spaced
    sin/cos features of t / T_max.
    """

    def __init__(self, data_size, hidden_arr, T_max, n_freq=50,
                 time_enc='scalar', rngs=nnx.Rngs(0)):
        self.T_max = T_max
        self.n_freq = n_freq
        self.time_enc = time_enc
        t_enc_dim = 2 * n_freq if time_enc == 'sinusoidal' else 1
        self.g2 = BasicMLP(2 * data_size + t_enc_dim, data_size, hidden_arr, rngs)

    def _t_enc(self, t):
        t_f = jnp.asarray(t, dtype=jnp.float32) / float(self.T_max)
        if self.time_enc == 'scalar':
            return jnp.atleast_1d(t_f)
        freqs = jnp.exp(jnp.linspace(0.0, jnp.log(float(self.T_max)), self.n_freq))
        return jnp.concatenate([jnp.sin(freqs * t_f), jnp.cos(freqs * t_f)])

    def step(self, x_t, z, t):
        """One-step transition sample x_{t+1} = g(x_t, z, t)."""
        return self.g2(jnp.concatenate([x_t, z, self._t_enc(t)], axis=-1)) + x_t

    def __call__(self, x_t, z, t):
        """Joint sample (x_t, x_{t+1})."""
        return jnp.concatenate([x_t, self.step(x_t, z, t)], axis=-1)


def load_model(run_dir, hidden=512, n_layers=2, T_max=1200, n_freq=50, time_enc='scalar',
               data_size=2):
    """Load ckpt_final.pkl and norm_stats.npy written by duffing.train."""
    model = BTMLP(data_size, [hidden] * n_layers, T_max, n_freq, time_enc, rngs=nnx.Rngs(0))
    with open(os.path.join(run_dir, 'ckpt_final.pkl'), 'rb') as f:
        nnx.update(model, pickle.load(f))
    norm = np.load(os.path.join(run_dir, 'norm_stats.npy'))
    return model, norm[0], norm[1]


def rollout(model, x0_norm, T, key):
    """Autoregressive rollout in normalized units: [N, d] -> [N, T+1, d]."""
    step = jax.jit(jax.vmap(model.step, in_axes=(0, 0, None)))
    x = jnp.asarray(x0_norm)
    xs = [np.array(x)]
    for t in range(T):
        key, sk = jax.random.split(key)
        x = step(x, jax.random.normal(sk, x.shape), jnp.array(t))
        xs.append(np.array(x))
    return np.stack(xs, axis=1)
