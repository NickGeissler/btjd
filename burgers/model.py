"""Latent block-triangular generator for Burgers.

The state is the flattened 16-dim autoencoder latent h (4 positions x 4
channels). g2 concatenates h_t and noise z channel-wise and applies FiLM
time-conditioned residual blocks with circular padding. The paper model
predicts h_{t+1} directly (residual=False).
"""

import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from burgers.autoencoder import Encoder1D, Decoder1D


def _sinusoidal_embedding(t, T_max, n_freq):
    t_f = jnp.asarray(t, dtype=jnp.float32) / float(T_max)
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(float(T_max)), n_freq))
    return jnp.concatenate([jnp.sin(freqs * t_f), jnp.cos(freqs * t_f)])


class _MLP(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs):
        self.fc1 = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.fc2 = nnx.Linear(out_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jax.nn.silu(self.fc1(x)))


class _FiLMResBlock1D(nnx.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, rngs):
        self.norm1 = nnx.LayerNorm(in_ch, rngs=rngs)
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3,), padding='CIRCULAR', rngs=rngs)
        self.film = nnx.Linear(time_emb_dim, 2 * out_ch, rngs=rngs)
        self.norm2 = nnx.LayerNorm(out_ch, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3,), padding='CIRCULAR', rngs=rngs)
        self.skip = (nnx.Conv(in_ch, out_ch, kernel_size=(1,), rngs=rngs)
                     if in_ch != out_ch else None)

    def __call__(self, x, t_emb):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        scale, shift = jnp.split(self.film(t_emb), 2)
        h = h * (1.0 + scale) + shift
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class _LatentG2(nnx.Module):
    def __init__(self, lat_ch, gen_ch, time_emb_dim, n_blocks, rngs, residual):
        self.residual = residual
        self.proj_in = nnx.Conv(2 * lat_ch, gen_ch, kernel_size=(1,), rngs=rngs)
        self.blocks = nnx.List([_FiLMResBlock1D(gen_ch, gen_ch, time_emb_dim, rngs)
                                for _ in range(n_blocks)])
        self.norm_out = nnx.LayerNorm(gen_ch, rngs=rngs)
        self.proj_out = nnx.Conv(gen_ch, lat_ch, kernel_size=(1,), rngs=rngs)

    def __call__(self, z, h_t, t_emb):
        x = jax.nn.silu(self.proj_in(jnp.concatenate([h_t, z], axis=-1)))
        for block in self.blocks:
            x = block(x, t_emb)
        out = self.proj_out(jax.nn.silu(self.norm_out(x)))
        return h_t + out if self.residual else out


class LatentBTNet(nnx.Module):
    """f(h_t, z, t) = (h_t, g(h_t, z, t)) on flattened latents of size lat_dim."""

    def __init__(self, seq_len=64, lat_ch=4, gen_ch=32, n_blocks=3, T_max=799, n_freq=50,
                 time_emb_dim=128, residual=False, rngs=nnx.Rngs(0)):
        self.seq_len = seq_len
        self.lat_ch = lat_ch
        self.lat_dim = (seq_len // 16) * lat_ch
        self.T_max = T_max
        self.n_freq = n_freq
        self.time_mlp = _MLP(2 * n_freq, time_emb_dim, rngs)
        self.g2_net = _LatentG2(lat_ch, gen_ch, time_emb_dim, n_blocks, rngs, residual)

    def _spatial(self, h):
        return h.reshape(self.seq_len // 16, self.lat_ch)

    def step(self, h_t, z, t):
        t_emb = self.time_mlp(_sinusoidal_embedding(t, self.T_max, self.n_freq))
        return self.g2_net(self._spatial(z), self._spatial(h_t), t_emb).reshape(-1)

    def __call__(self, h_t, z, t):
        return jnp.concatenate([h_t, self.step(h_t, z, t)])


def load_autoencoder(enc_dir, seq_len=64, enc_ch=16, lat_ch=4):
    enc = Encoder1D(seq_len, enc_ch, lat_ch, rngs=nnx.Rngs(0))
    dec = Decoder1D(seq_len, enc_ch, lat_ch, rngs=nnx.Rngs(0))
    with open(os.path.join(enc_dir, 'encoder.pkl'), 'rb') as f:
        nnx.update(enc, pickle.load(f))
    with open(os.path.join(enc_dir, 'decoder.pkl'), 'rb') as f:
        nnx.update(dec, pickle.load(f))
    norm = np.load(os.path.join(enc_dir, 'norm_stats.npy'))
    return enc, dec, norm[0], norm[1]


def load_generator(run_dir, seq_len=64, lat_ch=4, gen_ch=32, n_blocks=3, T_max=799,
                   residual=False):
    model = LatentBTNet(seq_len, lat_ch, gen_ch, n_blocks, T_max, residual=residual,
                        rngs=nnx.Rngs(0))
    with open(os.path.join(run_dir, 'ckpt_final.pkl'), 'rb') as f:
        nnx.update(model, pickle.load(f))
    return model


def rollout(model, enc, dec, x0_norm, T, key):
    """Encode x0, step in latent space, decode every step -> [N, T+1, 64] (normalized)."""
    N = x0_norm.shape[0]
    enc_jit = jax.jit(jax.vmap(enc))
    dec_jit = jax.jit(jax.vmap(dec))
    step = jax.jit(jax.vmap(model.step, in_axes=(0, 0, None)))
    h = enc_jit(jnp.array(x0_norm)).reshape(N, model.lat_dim)
    xs = [np.array(x0_norm)]
    for t in range(T):
        key, sk = jax.random.split(key)
        h = step(h, jax.random.normal(sk, (N, model.lat_dim)), jnp.array(t, dtype=jnp.int32))
        xs.append(np.array(dec_jit(h.reshape(N, model.seq_len // 16, model.lat_ch))))
    return np.stack(xs, axis=1)
