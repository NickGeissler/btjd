"""Evaluate a trained Duffing model (Table 1).

Rolls out one model trajectory from each test initial condition and reports

  * sliced W2 between generated and test samples at every time step (POT,
    200 projections), averaged over time, with its std over time;
  * the relative error of the trajectory QoI
        Q = E[ int phi(X) o dX ],  phi(x) = [psi(x1) tanh(x2), 0],
    psi the standard normal density, via the midpoint (Stratonovich) rule,
    with the paired standard error std(q_gen - q_gt) / sqrt(N) / |Q_gt|.

The paper computes W2 on the 5000-trajectory test sets and the QoI on
200k-trajectory pools (use --skip-w2 for the pools).
"""

import argparse
import json
import os

import jax
import numpy as np
import ot
from tqdm import tqdm

from duffing.model import load_model, rollout


def sliced_w2_over_time(pred, true, n_projections=200, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(pred.shape[1])
    for t in tqdm(range(pred.shape[1]), desc='sliced W2'):
        out[t] = float(ot.sliced_wasserstein_distance(
            pred[:, t], true[:, t], n_projections=n_projections, p=2,
            seed=rng.integers(0, 2 ** 31 - 1)))
    return out


def psi(x1, bandwidth=1.0):
    return np.exp(-x1 ** 2 / (2 * bandwidth ** 2)) / (np.sqrt(2 * np.pi) * bandwidth)


def qoi_per_traj(trajs, kappa=1.0, bandwidth=1.0, chunk=20000):
    """Midpoint rule for int psi(x1) tanh(kappa x2) dx1 along each trajectory."""
    out = np.empty(trajs.shape[0])
    for i in range(0, trajs.shape[0], chunk):
        seg = np.asarray(trajs[i:i + chunk], dtype=np.float64)
        x1, x2 = seg[:, :, 0], seg[:, :, 1]
        x1_mid = 0.5 * (x1[:, :-1] + x1[:, 1:])
        x2_mid = 0.5 * (x2[:, :-1] + x2[:, 1:])
        out[i:i + chunk] = np.sum(psi(x1_mid, bandwidth) * np.tanh(kappa * x2_mid)
                                  * np.diff(x1, axis=1), axis=1)
    return out


def qoi_stats(q_gen, q_gt):
    n = q_gt.shape[0]
    m_gen, m_gt = float(q_gen.mean()), float(q_gt.mean())
    rel_err = abs(m_gen - m_gt) / abs(m_gt)
    se_paired = float(np.std(q_gen - q_gt, ddof=1) / np.sqrt(n))
    se_unpaired = float(np.sqrt(q_gen.std() ** 2 + q_gt.std() ** 2) / np.sqrt(n))
    return dict(n=n, q_gt_mean=m_gt, q_gen_mean=m_gen, q_gt_std=float(q_gt.std()),
                q_gen_std=float(q_gen.std()), q_rel_err=rel_err,
                q_rel_err_se_paired=se_paired / abs(m_gt),
                q_rel_err_se_unpaired=se_unpaired / abs(m_gt))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--T', type=int, default=1200)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--time-enc', default='scalar', choices=['scalar', 'sinusoidal'])
    p.add_argument('--T-max', type=int, default=1200)
    p.add_argument('--n-projections', type=int, default=200)
    p.add_argument('--skip-w2', action='store_true')
    p.add_argument('--seed', type=int, default=1, help='rollout noise seed')
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    model, mean, std = load_model(a.run_dir, a.hidden, a.n_layers, a.T_max, time_enc=a.time_enc)
    gt = np.load(a.test_data).astype(np.float32)[:, :a.T + 1]
    x0 = (gt[:, 0] - mean) / std
    print(f'rolling out {gt.shape[0]} trajectories for T={a.T}', flush=True)
    gen = rollout(model, x0, a.T, jax.random.PRNGKey(a.seed)) * std + mean

    finite = np.isfinite(gen).all(axis=(1, 2))
    out = dict(run_dir=a.run_dir, test_data=a.test_data, T=a.T, seed=a.seed,
               n_trajectories=int(gt.shape[0]), n_diverged=int((~finite).sum()))
    gen, gt = gen[finite], gt[finite]

    if not a.skip_w2:
        w2 = sliced_w2_over_time(gen, gt, a.n_projections)
        out.update(sliced_w2_mean=float(w2.mean()), sliced_w2_std=float(w2.std()),
                   sliced_w2_per_t=w2.tolist())
        print(f'sliced W2: {w2.mean():.4e} (+- {w2.std():.4e})', flush=True)

    out.update(qoi_stats(qoi_per_traj(gen), qoi_per_traj(gt)))
    print(f'QoI rel err: {out["q_rel_err"]:.4e} (+- {out["q_rel_err_se_paired"]:.4e} paired SE)',
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {a.out_json}', flush=True)


if __name__ == '__main__':
    main()
