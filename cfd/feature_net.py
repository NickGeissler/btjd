"""Latent-space masked-autoencoder (MAE) feature network used by the CFD drift loss.

A ResNet encoder over the [16,16,4] latent (channels 256, 512) with a U-Net
decoder used only for masked-reconstruction pretraining. get_activations()
returns the multi-scale feature dictionary of the W-Flow / Drifting MAE
(per-location tokens, global and patch mean/std pools, norm_x); all_feature_keys
adds the flattened input as the 'global' key.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from cfd.autoencoder import ResidualBlock, Downsample, Upsample, _gn_groups


def _safe_std(x, axis, eps=1e-6):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axis)
    return jnp.sqrt(jnp.clip(var, min=0.0) + eps)


class _ResNetEncoder(nnx.Module):
    def __init__(self, in_channels, channels, blocks_per_stage, rngs):
        self.channels = channels
        c0 = channels[0]
        self.stem = nnx.Conv(in_channels, c0, kernel_size=(3, 3), padding='CIRCULAR',
                             use_bias=False, rngs=rngs)
        self.stem_gn = nnx.GroupNorm(c0, num_groups=_gn_groups(c0), rngs=rngs)
        downs, stages = [], []
        in_ch = c0
        for out_ch in channels:
            downs.append(Downsample(in_ch, out_ch, rngs))
            stages.append(nnx.List([ResidualBlock(out_ch, out_ch, rngs)
                                    for _ in range(blocks_per_stage)]))
            in_ch = out_ch
        self.downs = nnx.List(downs)
        self.stages = nnx.List(stages)

    def __call__(self, x, return_block_outputs=False):
        feats, block_outputs = {}, {}
        x = jax.nn.silu(self.stem_gn(self.stem(x)))
        feats['conv1'] = x
        for i, (down, blocks) in enumerate(zip(self.downs, self.stages)):
            x = down(x)
            outs = []
            for block in blocks:
                x = block(x)
                outs.append(x)
            block_outputs[f'layer{i + 1}'] = outs
            feats[f'layer{i + 1}'] = x
        return (feats, block_outputs) if return_block_outputs else feats


class _UpBlock(nnx.Module):
    def __init__(self, in_ch, skip_ch, out_ch, rngs):
        self.up = Upsample(in_ch, out_ch, rngs)
        self.refine = ResidualBlock(out_ch + skip_ch, out_ch, rngs)

    def __call__(self, x, skip):
        return self.refine(jnp.concatenate([self.up(x), skip], axis=-1))


class _UNetDecoderN(nnx.Module):
    def __init__(self, channels, out_channels, rngs):
        stage_channels = [channels[0]] + list(channels)
        self.bridge = ResidualBlock(stage_channels[-1], stage_channels[-1], rngs)
        self.ups = nnx.List([_UpBlock(stage_channels[i], stage_channels[i - 1],
                                      stage_channels[i - 1], rngs)
                             for i in range(len(stage_channels) - 1, 0, -1)])
        self.n_stages = len(channels)
        c0 = channels[0]
        self.head_norm = nnx.GroupNorm(c0, num_groups=_gn_groups(c0), rngs=rngs)
        self.head = nnx.Conv(c0, out_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, feats):
        x = self.bridge(feats[f'layer{self.n_stages}'])
        for i, up in enumerate(self.ups):
            src = self.n_stages - i
            x = up(x, feats['conv1' if src == 1 else f'layer{src - 1}'])
        return self.head(jax.nn.silu(self.head_norm(x)))


class MultiscaleFeatureNetLatent(nnx.Module):
    def __init__(self, in_channels=4, channels=(256, 512), blocks_per_stage=2, rngs=nnx.Rngs(0)):
        self.in_channels = in_channels
        self.channels = channels
        self.encoder = _ResNetEncoder(in_channels, channels, blocks_per_stage, rngs)
        self.decoder = _UNetDecoderN(channels, in_channels, rngs)

    def forward_pretrain(self, x_masked):
        return self.decoder(self.encoder(x_masked))

    def get_activations(self, x, patch_mean_size=(2, 4), patch_std_size=(2, 4), every_k_block=2):
        """x: [16,16,C] -> dict of [n_tokens, dim] feature sets."""
        out = {'norm_x': jnp.sqrt(jnp.mean(x ** 2, axis=(0, 1), keepdims=True) + 1e-6).reshape(1, -1)}
        feats, block_outputs = self.encoder(x, return_block_outputs=True)

        def process(name, feat):
            h, w, c = feat.shape
            out[name] = feat.reshape(h * w, c)
            out[f'{name}_mean'] = jnp.mean(feat, axis=(0, 1), keepdims=True).reshape(1, c)
            out[f'{name}_std'] = _safe_std(feat, axis=(0, 1)).reshape(1, c)
            for size, pool, tag in ([(s, 'mean', 'mean') for s in patch_mean_size]
                                    + [(s, 'std', 'std') for s in patch_std_size]):
                if h % size == 0 and w % size == 0:
                    r = feat.reshape(h // size, size, w // size, size, c).transpose(0, 2, 1, 3, 4)
                    r = r.reshape((h // size) * (w // size), size * size, c)
                    out[f'{name}_{tag}_{size}'] = (jnp.mean(r, axis=1) if pool == 'mean'
                                                   else _safe_std(r, axis=1))

        for name, feat in feats.items():
            process(name, feat)
        for i in range(1, len(self.channels) + 1):
            blocks = block_outputs.get(f'layer{i}', [])
            for k, feat in enumerate(blocks, start=1):
                if k % every_k_block == 0 and k != len(blocks):
                    process(f'layer{i}_blk{k}', feat)
        return out


def all_feature_keys(net, x):
    feats = net.get_activations(x)
    feats['global'] = x.reshape(1, -1)
    return feats
