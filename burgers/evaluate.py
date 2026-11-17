"""Evaluate a trained Burgers model (Table 3).

Rolls out one trajectory per test initial condition (encode -> latent steps ->
decode every step) and compares the ensemble means of the energy
E = 1/2 int u^2 dx and enstrophy Z = 1/2 int u_x^2 dx (periodic grid,
centered differences) against the test set. Reports the relative error
|mean_gen(t) - mean_gt(t)| / |mean_gt(t)| averaged over t = 0..T and its std
over t.
"""

import argparse
import json
import os

import jax
import numpy as np

from burgers.model import load_autoencoder, load_generator, rollout

DX = 1.0 / 64


def energy_enstrophy(trajs, dx=DX):
    """[M, T, Nx] -> per-trajectory energy and enstrophy [M, T]."""
    e = 0.5 * np.sum(trajs ** 2, axis=2) * dx
    u = np.pad(trajs, ((0, 0), (0, 0), (1, 1)), mode='wrap')
    u_x = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    return e, 0.5 * np.sum(u_x ** 2, axis=2) * dx


def relative_error_over_time(gen_stat, gt_stat):
    rel = np.abs(gen_stat.mean(axis=0) - gt_stat.mean(axis=0)) / np.abs(gt_stat.mean(axis=0))
    return float(rel.mean()), float(rel.std())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--enc-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--T', type=int, default=799)
    p.add_argument('--residual', action='store_true', help='must match training')
    p.add_argument('--seed', type=int, default=42, help='rollout noise seed')
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    enc, dec, mean, std = load_autoencoder(a.enc_dir)
    model = load_generator(a.run_dir, T_max=a.T, residual=a.residual)
    raw = np.load(a.test_data, mmap_mode='r').astype(np.float32)
    gt = np.array(raw[:, :a.T + 1])
    print(f'rolling out {gt.shape[0]} trajectories for T={a.T}', flush=True)
    gen = rollout(model, enc, dec, (gt[:, 0] - mean) / std, a.T, jax.random.PRNGKey(a.seed))
    gen = gen * std + mean

    e_gt, z_gt = energy_enstrophy(gt)
    e_gen, z_gen = energy_enstrophy(gen)
    e_mean, e_std = relative_error_over_time(e_gen, e_gt)
    z_mean, z_std = relative_error_over_time(z_gen, z_gt)
    out = dict(run_dir=a.run_dir, test_data=a.test_data, T=a.T, seed=a.seed,
               n_trajectories=int(gt.shape[0]),
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
