"""Evaluate the turbulence generator (Table 4).

Rolls out up to --rollout-len steps of 4 solver frames (249 steps for 1000
stored frames) from every latent test initial condition, decodes with the
frozen autoencoder, rescales to physical units and
compares the ensemble means of kinetic energy (1/2 mean |u|^2, u from the
streamfunction) and enstrophy (1/2 mean omega^2) with the test set. Reports
|mean_gen(t) - mean_gt(t)| / |mean_gt(t)| averaged over t = 1..T and its std
over t.
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from cfd.autoencoder import load_autoencoder
from cfd.generator import load_generator


def rollout_latent(model, z0, T, key, time_stride=4):
    """z0: [N, 16, 16, C] -> [N, T+1, 16, 16, C]."""
    step = jax.jit(jax.vmap(model.g2, in_axes=(0, 0, None)))
    h = jnp.array(z0)
    zs = [np.array(z0)]
    for t in range(T):
        key, sk = jax.random.split(key)
        h = step(jax.random.normal(sk, h.shape), h, jnp.float32(t * time_stride))
        zs.append(np.array(h))
    return np.stack(zs, axis=1)


def decode(ae, latents, batch_size=512):
    N, T1 = latents.shape[:2]
    flat = jnp.array(latents.reshape(N * T1, *latents.shape[2:]))
    dec = jax.jit(jax.vmap(ae.decode))
    out = [np.array(dec(flat[i:i + batch_size])) for i in range(0, flat.shape[0], batch_size)]
    return np.concatenate(out).reshape(N, T1, *out[0].shape[1:])


def _velocity(om):
    N = om.shape[0]
    kx = jnp.fft.fftfreq(N)[:, None] * N / (2 * jnp.pi) * jnp.ones((1, N // 2 + 1))
    ky = jnp.fft.rfftfreq(N)[None, :] * N / (2 * jnp.pi) * jnp.ones((N, 1))
    two_pi_i = 2j * jnp.pi
    lap = (two_pi_i ** 2 * (kx ** 2 + ky ** 2)).at[0, 0].set(1.0)
    psi_hat = -jnp.fft.rfftn(om) / lap
    return (jnp.fft.irfftn(two_pi_i * ky * psi_hat, s=(N, N)),
            jnp.fft.irfftn(-two_pi_i * kx * psi_hat, s=(N, N)))


def energy_enstrophy_means(trajs):
    """trajs: [M, T, H, W, 1] vorticity -> ensemble-mean energy and enstrophy [T]."""
    om = jnp.array(trajs[..., 0])

    def energy(o):
        vx, vy = _velocity(o)
        return 0.5 * jnp.mean(vx ** 2 + vy ** 2)

    e = np.array(jax.vmap(jax.vmap(energy))(om))
    z = np.array(0.5 * jnp.mean(om ** 2, axis=(-2, -1)))
    return e.mean(axis=0), z.mean(axis=0)


def relative_error(gen, gt):
    rel = np.abs(gen[1:] - gt[1:]) / (np.abs(gt[1:]) + 1e-12)
    return float(rel.mean()), float(rel.std())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ae-dir', required=True)
    p.add_argument('--latent-data', required=True, help='latent test set (initial conditions)')
    p.add_argument('--pixel-data', required=True, help='normalized vorticity test set')
    p.add_argument('--data-scale', type=float, required=True,
                   help='the training normalization constant (*_scale.npy)')
    p.add_argument('--out-json', required=True)
    p.add_argument('--rollout-len', type=int, default=250)
    p.add_argument('--time-stride', type=int, default=4)
    p.add_argument('--n-traj', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    lat = np.load(a.latent_data, mmap_mode='r')
    pix = np.load(a.pixel_data, mmap_mode='r')
    n = min(a.n_traj, lat.shape[0])
    s = a.time_stride
    T = min(a.rollout_len, (pix.shape[1] - 1) // s)     # 249 steps for 1000 stored frames
    gt = np.array(pix[:n, :T * s + 1:s]) * a.data_scale
    model, ae = load_generator(a.ckpt), load_autoencoder(a.ae_dir)
    print(f'rolling out {n} trajectories for {T} steps', flush=True)
    gen = decode(ae, rollout_latent(model, np.array(lat[:n, 0]), T, jax.random.PRNGKey(a.seed), s))
    gen = gen * a.data_scale

    e_gen, z_gen = energy_enstrophy_means(gen)
    e_gt, z_gt = energy_enstrophy_means(gt)
    e_mean, e_std = relative_error(e_gen, e_gt)
    z_mean, z_std = relative_error(z_gen, z_gt)
    out = dict(ckpt=a.ckpt, n_trajectories=n, rollout_len=T, time_stride=s, seed=a.seed,
               energy_mean_relerr=e_mean, energy_std_relerr=e_std,
               enstrophy_mean_relerr=z_mean, enstrophy_std_relerr=z_std)
    print(f'energy rel err {e_mean:.3e} (+- {e_std:.3e})   '
          f'enstrophy rel err {z_mean:.3e} (+- {z_std:.3e})', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {a.out_json}', flush=True)


if __name__ == '__main__':
    main()
