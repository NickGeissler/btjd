"""Rayleigh-Benard marginal histograms at selected rollout times (Figure: RB marginals).

Writes one panel per (component, time step) to <out-dir>/rb_histograms/:
BTJD as a filled histogram, ground truth as an outline, with bin edges shared
across time for each component. Rollouts are cached to <out-dir>/rb_rollout.npz.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax

from common.plot_style import C_GT, C_BT, use_paper_style, figsize
from rb.model import load_model, rollout

COMPONENTS = [0, 1, 3, 7, 8]     # X1, X2, X4, X8, X9
N_BINS = 50


def make_panels(gt, gen, steps, out_dir, panel_fraction=0.24, panel_aspect=0.95):
    """gt, gen: [N, len(steps), 9], already restricted to the plotted steps."""
    use_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    for comp in COMPONENTS:
        lo, hi = np.inf, -np.inf
        for k in range(len(steps)):
            vals = np.concatenate([gt[:, k, comp], gen[:, k, comp]])
            lo = min(lo, float(np.nanpercentile(vals, 0.2)))
            hi = max(hi, float(np.nanpercentile(vals, 99.8)))
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        edges = np.linspace(lo - pad, hi + pad, N_BINS + 1)
        for k, t in enumerate(steps):
            fig, ax = plt.subplots(figsize=figsize(panel_fraction, aspect=panel_aspect))
            ax.hist(gen[:, k, comp], bins=edges, density=True, color=C_BT, alpha=0.45,
                    edgecolor='none')
            ax.hist(gt[:, k, comp], bins=edges, density=True, histtype='step', linewidth=0.4,
                    color=C_GT, edgecolor=C_GT)
            ax.tick_params(axis='y', labelleft=False, left=False)
            ax.set_xlim(edges[0], edges[-1])
            stem = os.path.join(out_dir, f'rb_hist_C{comp + 1}_t{t}')
            for ext in ('pdf', 'png'):
                fig.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close(fig)
            print(f'wrote {stem}.pdf', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--mu', type=float, default=13.65)
    p.add_argument('--n-particles', type=int, default=20000)
    p.add_argument('--steps', type=int, nargs='+', default=[0, 500, 1000, 1500, 2000])
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--recompute', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, 'rb_rollout.npz')
    if os.path.exists(cache) and not a.recompute:
        z = np.load(cache)
        gt, gen = z['gt'], z['gen']
    else:
        model, mean, std = load_model(a.run_dir, a.hidden, a.n_layers)
        T = max(a.steps)
        gt = np.load(a.test_data).astype(np.float32)[:a.n_particles, :T + 1]
        gen = rollout(model, (gt[:, 0] - mean) / std, T, a.mu, jax.random.PRNGKey(a.seed))
        gen = gen * std + mean
        gt, gen = gt[:, a.steps], gen[:, a.steps]
        np.savez_compressed(cache, gt=gt, gen=gen, steps=np.array(a.steps))
    make_panels(gt, gen, a.steps, os.path.join(a.out_dir, 'rb_histograms'))


if __name__ == '__main__':
    main()
