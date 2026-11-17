"""1D convolutional autoencoder for Burgers: u in R^64 -> latent [4, lat_ch] -> R^64.

Pre-activation residual blocks (GroupNorm -> SiLU -> conv, twice), four
stride-2 stages, circular padding, no encoder-decoder skip connections.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _gn_groups(num_channels, max_groups=32, min_channels_per_group=4):
    """Largest divisor of num_channels <= max_groups with >= min_channels_per_group per group."""
    g = min(max_groups, num_channels // min_channels_per_group, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return max(g, 1)


def _group_norm(num_channels, rngs):
    return nnx.GroupNorm(num_channels, num_groups=_gn_groups(num_channels), rngs=rngs)


def _gn(norm, x):
    """GroupNorm on an unbatched [L, C] input (adds a dummy batch axis)."""
    return norm(x[None])[0]


class ResidualBlock1D(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs):
        self.norm1 = _group_norm(in_ch, rngs)
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3,), padding='CIRCULAR',
                              use_bias=False, rngs=rngs)
        self.norm2 = _group_norm(out_ch, rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3,), padding='CIRCULAR',
                              use_bias=False, rngs=rngs)
        self.skip = (nnx.Conv(in_ch, out_ch, kernel_size=(1,), use_bias=False, rngs=rngs)
                     if in_ch != out_ch else None)

    def __call__(self, x):
        h = self.conv1(jax.nn.silu(_gn(self.norm1, x)))
        h = self.conv2(jax.nn.silu(_gn(self.norm2, h)))
        return h + (self.skip(x) if self.skip is not None else x)


class Downsample1D(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=(3,), strides=(2,),
                             padding='CIRCULAR', rngs=rngs)

    def __call__(self, x):
        return jax.nn.silu(self.conv(x))


class Upsample1D(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=(3,), padding='CIRCULAR', rngs=rngs)

    def __call__(self, x):
        return jax.nn.silu(self.conv(jnp.repeat(x, 2, axis=0)))


class Encoder1D(nnx.Module):
    """[64] -> [4, lat_ch]."""

    def __init__(self, seq_len=64, enc_ch=16, lat_ch=4, blocks_per_stage=2,
                 bottleneck_blocks=2, rngs=nnx.Rngs(0)):
        assert seq_len % 16 == 0
        c0, c1 = enc_ch, enc_ch * 2
        self.stem = nnx.Conv(1, c0, kernel_size=(3,), padding='CIRCULAR', rngs=rngs)
        self.stage1 = nnx.List([ResidualBlock1D(c0, c0, rngs) for _ in range(blocks_per_stage)])
        self.down1 = Downsample1D(c0, c0, rngs)
        self.stage2 = nnx.List([ResidualBlock1D(c0, c1, rngs)]
                               + [ResidualBlock1D(c1, c1, rngs) for _ in range(blocks_per_stage - 1)])
        self.down2 = Downsample1D(c1, c1, rngs)
        self.stage3 = nnx.List([ResidualBlock1D(c1, c1, rngs) for _ in range(blocks_per_stage)])
        self.down3 = Downsample1D(c1, c1, rngs)
        self.stage4 = nnx.List([ResidualBlock1D(c1, c0, rngs)]
                               + [ResidualBlock1D(c0, c0, rngs) for _ in range(blocks_per_stage - 1)])
        self.down4 = Downsample1D(c0, c0, rngs)
        self.bottleneck = nnx.List([ResidualBlock1D(c0, c0, rngs) for _ in range(bottleneck_blocks)])
        self.head_norm = _group_norm(c0, rngs)
        self.proj_out = nnx.Conv(c0, lat_ch, kernel_size=(1,), rngs=rngs)

    def __call__(self, x):
        h = self.stem(x[:, None])
        for stage, down in ((self.stage1, self.down1), (self.stage2, self.down2),
                            (self.stage3, self.down3), (self.stage4, self.down4)):
            for block in stage:
                h = block(h)
            h = down(h)
        for block in self.bottleneck:
            h = block(h)
        return self.proj_out(jax.nn.silu(_gn(self.head_norm, h)))


class Decoder1D(nnx.Module):
    """[4, lat_ch] -> [64]."""

    def __init__(self, seq_len=64, enc_ch=16, lat_ch=4, blocks_per_stage=2,
                 bottleneck_blocks=2, rngs=nnx.Rngs(0)):
        assert seq_len % 16 == 0
        c0, c1 = enc_ch, enc_ch * 2
        self.proj_in = nnx.Conv(lat_ch, c0, kernel_size=(1,), rngs=rngs)
        self.bottleneck = nnx.List([ResidualBlock1D(c0, c0, rngs) for _ in range(bottleneck_blocks)])
        self.up1 = Upsample1D(c0, c1, rngs)
        self.stage1 = nnx.List([ResidualBlock1D(c1, c1, rngs) for _ in range(blocks_per_stage - 1)])
        self.up2 = Upsample1D(c1, c1, rngs)
        self.stage2 = nnx.List([ResidualBlock1D(c1, c1, rngs) for _ in range(blocks_per_stage - 1)])
        self.up3 = Upsample1D(c1, c0, rngs)
        self.stage3 = nnx.List([ResidualBlock1D(c0, c0, rngs) for _ in range(blocks_per_stage - 1)])
        self.up4 = Upsample1D(c0, c0, rngs)
        self.stage4 = nnx.List([ResidualBlock1D(c0, c0, rngs) for _ in range(blocks_per_stage - 1)])
        self.head_norm = _group_norm(c0, rngs)
        self.head = nnx.Conv(c0, 1, kernel_size=(1,), rngs=rngs)

    def __call__(self, z):
        x = self.proj_in(z)
        for block in self.bottleneck:
            x = block(x)
        for up, stage in ((self.up1, self.stage1), (self.up2, self.stage2),
                          (self.up3, self.stage3), (self.up4, self.stage4)):
            x = up(x)
            for block in stage:
                x = block(x)
        return self.head(jax.nn.silu(_gn(self.head_norm, x)))[..., 0]
