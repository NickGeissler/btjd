"""Convolutional autoencoder for the turbulence fields: [64,64,1] -> [16,16,4] -> [64,64,1].

Pre-activation residual blocks (GroupNorm -> SiLU -> conv, twice), two
stride-2 stages, circular padding, no encoder-decoder skip connections.
"""

import os
import pickle

import jax
import jax.numpy as jnp
from flax import nnx


def _gn_groups(num_channels, max_groups=32):
    g = min(max_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return max(g, 1)


class ResidualBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs, stride=1):
        self.norm1 = nnx.GroupNorm(in_ch, num_groups=_gn_groups(in_ch), rngs=rngs)
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), strides=(stride, stride),
                              padding='CIRCULAR', use_bias=False, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=_gn_groups(out_ch), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding='CIRCULAR',
                              use_bias=False, rngs=rngs)
        need_proj = (in_ch != out_ch) or (stride != 1)
        self.skip = (nnx.Conv(in_ch, out_ch, kernel_size=(1, 1), strides=(stride, stride),
                              use_bias=False, rngs=rngs) if need_proj else None)

    def __call__(self, x):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class Downsample(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), strides=(2, 2),
                             padding='CIRCULAR', rngs=rngs)

    def __call__(self, x):
        return jax.nn.silu(self.conv(x))


class Upsample(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding='CIRCULAR', rngs=rngs)

    def __call__(self, x):
        return jax.nn.silu(self.conv(jnp.repeat(jnp.repeat(x, 2, axis=0), 2, axis=1)))


class Encoder(nnx.Module):
    def __init__(self, in_channels=1, channels=(96, 192), latent_channels=4,
                 blocks_per_stage=2, bottleneck_blocks=2, rngs=nnx.Rngs(0)):
        c0, c1 = channels
        self.stem = nnx.Conv(in_channels, c0, kernel_size=(3, 3), padding='CIRCULAR', rngs=rngs)
        self.stage1 = nnx.List([ResidualBlock(c0, c0, rngs) for _ in range(blocks_per_stage)])
        self.down1 = Downsample(c0, c0, rngs)
        self.stage2 = nnx.List([ResidualBlock(c0, c1, rngs)]
                               + [ResidualBlock(c1, c1, rngs) for _ in range(1, blocks_per_stage)])
        self.down2 = Downsample(c1, c1, rngs)
        self.bottleneck = nnx.List([ResidualBlock(c1, c1, rngs) for _ in range(bottleneck_blocks)])
        self.head_norm = nnx.GroupNorm(c1, num_groups=_gn_groups(c1), rngs=rngs)
        self.proj_out = nnx.Conv(c1, latent_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        x = self.stem(x)
        for block in self.stage1:
            x = block(x)
        x = self.down1(x)
        for block in self.stage2:
            x = block(x)
        x = self.down2(x)
        for block in self.bottleneck:
            x = block(x)
        return self.proj_out(jax.nn.silu(self.head_norm(x)))


class Decoder(nnx.Module):
    def __init__(self, out_channels=1, channels=(96, 192), latent_channels=4,
                 blocks_per_stage=2, bottleneck_blocks=2, rngs=nnx.Rngs(0)):
        c0, c1 = channels
        self.proj_in = nnx.Conv(latent_channels, c1, kernel_size=(1, 1), rngs=rngs)
        self.bottleneck = nnx.List([ResidualBlock(c1, c1, rngs) for _ in range(bottleneck_blocks)])
        self.up1 = Upsample(c1, c0, rngs)
        self.stage1 = nnx.List([ResidualBlock(c0, c0, rngs) for _ in range(blocks_per_stage - 1)])
        self.up2 = Upsample(c0, c0, rngs)
        self.stage2 = nnx.List([ResidualBlock(c0, c0, rngs) for _ in range(blocks_per_stage - 1)])
        self.head_norm = nnx.GroupNorm(c0, num_groups=_gn_groups(c0), rngs=rngs)
        self.head = nnx.Conv(c0, out_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, z):
        x = self.proj_in(z)
        for block in self.bottleneck:
            x = block(x)
        x = self.up1(x)
        for block in self.stage1:
            x = block(x)
        x = self.up2(x)
        for block in self.stage2:
            x = block(x)
        return self.head(jax.nn.silu(self.head_norm(x)))


class ConvAutoencoder(nnx.Module):
    def __init__(self, in_channels=1, channels=(96, 192), latent_channels=4,
                 blocks_per_stage=2, bottleneck_blocks=2, rngs=nnx.Rngs(0)):
        self.encoder = Encoder(in_channels, channels, latent_channels, blocks_per_stage,
                               bottleneck_blocks, rngs)
        self.decoder = Decoder(in_channels, channels, latent_channels, blocks_per_stage,
                               bottleneck_blocks, rngs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


def load_autoencoder(ae_dir, ema=True):
    """Load the (EMA) autoencoder written by cfd.train_autoencoder."""
    ae = ConvAutoencoder(rngs=nnx.Rngs(0))
    suffix = '_ema' if ema else ''
    with open(os.path.join(ae_dir, f'encoder{suffix}.pkl'), 'rb') as f:
        nnx.update(ae.encoder, pickle.load(f))
    with open(os.path.join(ae_dir, f'decoder{suffix}.pkl'), 'rb') as f:
        nnx.update(ae.decoder, pickle.load(f))
    return ae
