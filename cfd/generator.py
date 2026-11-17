"""Latent block-triangular generator for 2D turbulence: a FiLM time-conditioned
U-Net over [16,16,4] latents (channels 256, 512) with input [h_t | z] and
residual output h_{t+1} = h_t + UNet(...)."""

import pickle

import jax
import jax.numpy as jnp
from flax import nnx

from cfd.autoencoder import _gn_groups


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


class _FiLMResBlock2D(nnx.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, rngs):
        self.norm1 = nnx.GroupNorm(in_ch, num_groups=_gn_groups(in_ch), rngs=rngs)
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding='CIRCULAR', rngs=rngs)
        self.film = nnx.Linear(time_emb_dim, 2 * out_ch, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=_gn_groups(out_ch), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding='CIRCULAR', rngs=rngs)
        self.skip = (nnx.Conv(in_ch, out_ch, kernel_size=(1, 1), rngs=rngs)
                     if in_ch != out_ch else None)

    def __call__(self, x, t_emb):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        scale, shift = jnp.split(self.film(t_emb), 2)
        h = self.conv2(jax.nn.silu(self.norm2(h * (1.0 + scale) + shift)))
        return h + (self.skip(x) if self.skip is not None else x)


class UNet2DLatent(nnx.Module):
    def __init__(self, in_channels=4, channels=(256, 512), time_emb_dim=128, T_max=999,
                 n_freq=50, rngs=nnx.Rngs(0)):
        self.T_max = T_max
        self.n_freq = n_freq
        self.time_mlp = _MLP(2 * n_freq, time_emb_dim, rngs)
        c0 = channels[0]
        self.stem = nnx.Conv(2 * in_channels, c0, kernel_size=(3, 3), padding='CIRCULAR', rngs=rngs)
        stage = [c0] + list(channels)
        encs, downs = [], []
        for i in range(len(channels)):
            encs.append(_FiLMResBlock2D(stage[i], stage[i], time_emb_dim, rngs))
            downs.append(nnx.Conv(stage[i], stage[i + 1], kernel_size=(3, 3), strides=(2, 2),
                                  padding='CIRCULAR', rngs=rngs))
        self.encs = nnx.List(encs)
        self.downs = nnx.List(downs)
        self.bottleneck = _FiLMResBlock2D(stage[-1], stage[-1], time_emb_dim, rngs)
        up_convs, decs = [], []
        for i in range(len(channels) - 1, -1, -1):
            up_convs.append(nnx.Conv(stage[i + 1], stage[i], kernel_size=(3, 3),
                                     padding='CIRCULAR', rngs=rngs))
            decs.append(_FiLMResBlock2D(2 * stage[i], stage[i], time_emb_dim, rngs))
        self.up_convs = nnx.List(up_convs)
        self.decs = nnx.List(decs)
        self.head_norm = nnx.GroupNorm(c0, num_groups=_gn_groups(c0), rngs=rngs)
        self.head = nnx.Conv(c0, in_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, z, h_t, t):
        t_emb = self.time_mlp(_sinusoidal_embedding(t, self.T_max, self.n_freq))
        x = jax.nn.silu(self.stem(jnp.concatenate([h_t, z], axis=-1)))
        skips = []
        for enc, down in zip(self.encs, self.downs):
            s = enc(x, t_emb)
            skips.append(s)
            x = jax.nn.silu(down(s))
        x = self.bottleneck(x, t_emb)
        for up_conv, dec, skip in zip(self.up_convs, self.decs, reversed(skips)):
            x = jax.nn.silu(up_conv(jnp.repeat(jnp.repeat(x, 2, axis=0), 2, axis=1)))
            x = dec(jnp.concatenate([x, skip], axis=-1), t_emb)
        return h_t + self.head(jax.nn.silu(self.head_norm(x)))


class BlockTriangularUNet2DLatent(nnx.Module):
    """f(h_t, z, t) = (h_t, g2(z, h_t, t)), concatenated along channels."""

    def __init__(self, in_channels=4, channels=(256, 512), time_emb_dim=128, rngs=nnx.Rngs(0)):
        self.g2 = UNet2DLatent(in_channels, channels, time_emb_dim, rngs=rngs)

    def __call__(self, h_t, z, t):
        return jnp.concatenate([h_t, self.g2(z, h_t, t)], axis=-1)


def load_generator(ckpt):
    model = BlockTriangularUNet2DLatent(rngs=nnx.Rngs(0))
    with open(ckpt, 'rb') as f:
        nnx.update(model, pickle.load(f))
    return model
