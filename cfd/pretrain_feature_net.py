"""Masked-autoencoder pretraining of the latent feature network.

Random latent pixels (fraction --mask-ratio) are zeroed and the network
reconstructs them; the loss is the MSE on the masked region. Frames are taken
on the stride-4 time grid used by the generator. Writes encoder.pkl,
decoder.pkl and _ema versions; the generator uses encoder_ema.pkl.

As for the autoencoder, --test-data only feeds held-out loss logging but draws
from the random stream, so it must be given to reproduce the paper weights.

    python -m cfd.pretrain_feature_net --data data/cfd/stochasticJCFDTrain_latent.npy \
        --test-data data/cfd/stochasticJCFDTest_latent.npy --out-dir runs/cfd/mae
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

from cfd.feature_net import MultiscaleFeatureNetLatent


def make_patch_mask(key, mask_ratio, img_size, patch_size):
    """mask_ratio: [B] -> mask [B, H, W, 1], 1 = masked."""
    n = img_size // patch_size
    mask = (jax.random.uniform(key, (mask_ratio.shape[0], n, n))
            < mask_ratio[:, None, None]).astype(jnp.float32)
    mask = jnp.repeat(jnp.repeat(mask, patch_size, axis=1), patch_size, axis=2)
    return mask[..., None]


def masked_mse(x_hat, x, mask):
    return (jnp.sum((x_hat - x) ** 2 * mask, axis=(1, 2, 3))
            / (jnp.sum(mask, axis=(1, 2, 3)) + 1e-8))


@nnx.jit
def train_step(net, optimizer, x, mask):
    def loss_fn(net):
        per_sample = masked_mse(jax.vmap(net.forward_pretrain)(x * (1.0 - mask)), x, mask)
        return jnp.mean(per_sample), per_sample

    (_, per_sample), grads = nnx.value_and_grad(loss_fn, has_aux=True)(net)
    optimizer.update(net, grads)
    return per_sample


@nnx.jit
def eval_step(net, x, mask):
    return masked_mse(jax.vmap(net.forward_pretrain)(x * (1.0 - mask)), x, mask)


def load_frames(path, time_stride):
    raw = np.load(path, mmap_mode='r')[:, ::time_stride]
    return raw[:].reshape(-1, *raw.shape[2:]).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--test-data', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--time-stride', type=int, default=4)
    p.add_argument('--mask-ratio', type=float, default=0.25)
    p.add_argument('--patch-size', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--n-steps', type=int, default=100_000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=4000)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--eval-samples', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)
    frames = load_frames(a.data, a.time_stride)
    test_frames = load_frames(a.test_data, a.time_stride) if a.test_data else None
    img_size, C = frames.shape[1], frames.shape[-1]

    net = MultiscaleFeatureNetLatent(in_channels=C, rngs=nnx.Rngs(a.seed))
    schedule = optax.join_schedules([optax.linear_schedule(0.0, a.lr, a.warmup_steps),
                                     optax.constant_schedule(a.lr)], boundaries=[a.warmup_steps])
    tx = optax.chain(optax.clip_by_global_norm(2.0),
                     optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=a.weight_decay))
    optimizer = nnx.Optimizer(net, tx, wrt=nnx.Param)
    ema = jax.tree.map(jnp.array, nnx.state(net, nnx.Param))

    @nnx.jit
    def ema_update(ema, net, decay):
        return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p,
                            ema, nnx.state(net, nnx.Param))

    def ratios(key, n):
        return jax.random.uniform(key, (n,), minval=a.mask_ratio, maxval=a.mask_ratio)

    csv_file = open(os.path.join(a.out_dir, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'test_loss', 'elapsed_s'])
    key = jax.random.PRNGKey(a.seed)
    t0 = time.time()
    for step in range(a.n_steps):
        key, k_idx, k_ratio, k_mask = jax.random.split(key, 4)
        idx = np.array(jax.random.randint(k_idx, (a.batch_size,), 0, frames.shape[0]))
        mask = make_patch_mask(k_mask, ratios(k_ratio, a.batch_size), img_size, a.patch_size)
        loss = train_step(net, optimizer, jnp.array(frames[idx]), mask)
        ema = ema_update(ema, net, a.ema_decay)
        if step % a.log_every == 0:
            test_loss = ''
            if test_frames is not None and step % a.eval_every == 0:
                key, k_i, k_r, k_m = jax.random.split(key, 4)
                n = min(a.eval_samples, test_frames.shape[0])
                t_idx = np.array(jax.random.randint(k_i, (n,), 0, test_frames.shape[0]))
                t_mask = make_patch_mask(k_m, ratios(k_r, n), img_size, a.patch_size)
                test_loss = float(np.mean(eval_step(net, jnp.array(test_frames[t_idx]), t_mask)))
            elapsed = time.time() - t0
            print(f'step {step:6d}  loss={float(np.mean(loss)):.4e}  test={test_loss}  '
                  f'elapsed={elapsed:.0f}s', flush=True)
            writer.writerow([step, float(np.mean(loss)), test_loss, round(elapsed, 1)])
            csv_file.flush()
    csv_file.close()

    for suffix, state in (('', nnx.state(net, nnx.Param)), ('_ema', ema)):
        full = jax.tree.map(np.array, state)
        for part in ('encoder', 'decoder'):
            with open(os.path.join(a.out_dir, f'{part}{suffix}.pkl'), 'wb') as f:
                pickle.dump(full[part], f)
    print(f'saved feature network (+ _ema) -> {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
