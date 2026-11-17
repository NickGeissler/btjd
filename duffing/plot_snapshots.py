"""Duffing phase-space snapshots, ground truth vs BTJD (Figure: phase snapshots).

Rolls out one model trajectory from each of --n-particles test initial
conditions and writes one PDF/PNG panel per time step. Both clouds use the
same marker size and alpha and are drawn in a single shuffled scatter so
neither covers the other. Rollouts are cached to <out-dir>/snapshots_<tag>.npz.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import jax

from common.plot_style import C_GT, C_BT, use_paper_style, figsize, scale_panel_fonts
from duffing.model import load_model, rollout


def make_panels(gt, gen, steps, out_stem, dt, point_size=6.0, alpha=0.45,
                div_thresh=20.0, panel_fraction=0.24, panel_aspect=0.95):
    ok = np.isfinite(gen).all(axis=(1, 2)) & (np.abs(gen).max(axis=(1, 2)) < div_thresh)
    gen = gen[ok]
    xlim = 1.05 * np.abs(np.nanquantile(gt[..., 0], [0.001, 0.999])).max()
    ylim = 1.05 * np.abs(np.nanquantile(gt[..., 1], [0.001, 0.999])).max()
    point_size *= (panel_fraction / 0.48) ** 2

    use_paper_style()
    scale_panel_fonts(panel_fraction)
    for t in steps:
        fig, ax = plt.subplots(figsize=figsize(panel_fraction, aspect=panel_aspect))
        pts = np.concatenate([gt[:, t, :2], gen[:, t, :2]], axis=0)
        cols = np.array([C_GT] * len(gt) + [C_BT] * len(gen))
        order = np.random.default_rng(0).permutation(len(pts))
        ax.scatter(pts[order, 0], pts[order, 1], s=point_size, alpha=alpha,
                   color=cols[order], linewidths=0, rasterized=True, zorder=2)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel('$x_1$', labelpad=1)
        ax.set_ylabel('$x_2$', labelpad=0)
        ax.tick_params(pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out_stem}_step{t}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f'wrote {out_stem}_step{t}.pdf  [tau={t * dt:g}]', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[300, 600, 900, 1200])
    p.add_argument('--n-particles', type=int, default=2000)
    p.add_argument('--dt', type=float, default=0.01, help='only used for labels')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--time-enc', default='scalar', choices=['scalar', 'sinusoidal'])
    p.add_argument('--T-max', type=int, default=1200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tag', default='paper')
    p.add_argument('--recompute', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, f'snapshots_{a.tag}.npz')
    if os.path.exists(cache) and not a.recompute:
        z = np.load(cache)
        gt, gen = z['gt'], z['gen']
    else:
        model, mean, std = load_model(a.run_dir, a.hidden, a.n_layers, a.T_max,
                                      time_enc=a.time_enc)
        test = np.load(a.test_data).astype(np.float32)
        T = max(a.steps)
        idx = np.random.default_rng(a.seed).choice(test.shape[0], min(a.n_particles, test.shape[0]),
                                                   replace=False)
        gt = test[idx, :T + 1]
        gen = rollout(model, (gt[:, 0] - mean) / std, T, jax.random.PRNGKey(a.seed)) * std + mean
        np.savez_compressed(cache, gt=gt, gen=gen, T=T)
    make_panels(gt, gen, a.steps, os.path.join(a.out_dir, f'phase_snapshots_{a.tag}'), a.dt)


if __name__ == '__main__':
    main()
