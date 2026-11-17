"""Evaluate a trained Rayleigh-Benard model at an unseen mu (Table 2).

Rolls out one model trajectory from each test initial condition and reports

  * the debiased Sinkhorn divergence (ott-jax, epsilon=1e-3) between generated
    and test samples at every time step, computed in normalized coordinates,
    averaged over time, with its std over time;
  * the relative error of the circulation QoI in the (x6, x7) plane,
        Q = E[ int phi(X) o dX ],  phi(x) = Q^T R Q x,
    via the midpoint (Stratonovich) rule, with the paired standard error
    std(q_gen - q_gt) / sqrt(N) / |Q_gt|.
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from tqdm import tqdm

from rb.model import load_model, rollout


def sinkhorn_divergence_over_time(pred, true, epsilon=1e-3, t_subsample=1, max_samples=20000):
    @jax.jit
    def div(x, y):
        return sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud, x, y, epsilon=epsilon)[0]

    rng = np.random.RandomState(0)
    n_pred, n_true = min(max_samples, pred.shape[0]), min(max_samples, true.shape[0])
    vals = []
    for t in tqdm(range(0, pred.shape[1], t_subsample), desc='sinkhorn'):
        x = jnp.array(pred[rng.choice(pred.shape[0], n_pred, replace=False), t])
        y = jnp.array(true[rng.choice(true.shape[0], n_true, replace=False), t])
        vals.append(float(div(x, y)))
    return np.array(vals)


def circulation_per_traj(trajs, i=5, j=6):
    """Midpoint rule for int (-x_j dx_i + x_i dx_j) in the (i, j) plane (0-indexed)."""
    xi, xj = trajs[:, :, i], trajs[:, :, j]
    xi_mid = (xi[:, :-1] + xi[:, 1:]) / 2
    xj_mid = (xj[:, :-1] + xj[:, 1:]) / 2
    return np.sum(-xj_mid * np.diff(xi, axis=1) + xi_mid * np.diff(xj, axis=1), axis=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--mu', type=float, default=13.65)
    p.add_argument('--T', type=int, default=2000)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--epsilon', type=float, default=1e-3)
    p.add_argument('--t-subsample', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=20000)
    p.add_argument('--skip-sinkhorn', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='rollout noise seed')
    return p.parse_args()


def main():
    a = parse_args()
    print(f'JAX devices: {jax.devices()}', flush=True)
    model, mean, std = load_model(a.run_dir, a.hidden, a.n_layers)
    gt = np.load(a.test_data).astype(np.float32)[:, :a.T + 1]
    gt_norm = (gt - mean) / std
    print(f'rolling out {gt.shape[0]} trajectories for T={a.T} at mu={a.mu}', flush=True)
    gen_norm = rollout(model, gt_norm[:, 0], a.T, a.mu, jax.random.PRNGKey(a.seed))
    gen = gen_norm * std + mean

    out = dict(run_dir=a.run_dir, test_data=a.test_data, mu=a.mu, T=a.T, seed=a.seed,
               n_trajectories=int(gt.shape[0]))
    if not a.skip_sinkhorn:
        sd = sinkhorn_divergence_over_time(gen_norm, gt_norm, a.epsilon, a.t_subsample,
                                           a.max_samples)
        out.update(sinkhorn_epsilon=a.epsilon, sinkhorn_mean=float(sd.mean()),
                   sinkhorn_std=float(sd.std()), sinkhorn_per_t=sd.tolist())
        print(f'Sinkhorn divergence: {sd.mean():.4e} (+- {sd.std():.4e})', flush=True)

    q_gt, q_gen = circulation_per_traj(gt), circulation_per_traj(gen)
    n = q_gt.shape[0]
    m_gt, m_gen = float(q_gt.mean()), float(q_gen.mean())
    se_paired = float(np.std(q_gen - q_gt, ddof=1) / np.sqrt(n))
    se_unpaired = float(np.hypot(q_gt.std(), q_gen.std()) / np.sqrt(n))
    out.update(q_gt_mean=m_gt, q_gen_mean=m_gen, q_rel_err=abs(m_gen - m_gt) / abs(m_gt),
               q_rel_err_se_paired=se_paired / abs(m_gt),
               q_rel_err_se_unpaired=se_unpaired / abs(m_gt))
    print(f'QoI rel err: {out["q_rel_err"]:.4e} (+- {out["q_rel_err_se_paired"]:.4e} paired SE)',
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {a.out_json}', flush=True)


if __name__ == '__main__':
    main()
