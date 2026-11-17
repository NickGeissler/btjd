"""Burgers solution snapshots, ground truth vs BTJD (Figure: Burgers snapshots).

Rolls out one trajectory from each of --n-particles test initial conditions
and writes one panel per step with --n-curves curves per ensemble, drawn in a
single shuffled collection so neither ensemble covers the other. Rollouts are
cached to <out-dir>/burgers_snapshots_<tag>.npz.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
import jax

from common.plot_style import C_GT, C_BT, use_paper_style, figsize, scale_panel_fonts
from burgers.model import load_autoencoder, load_generator, rollout

DX = 1.0 / 64
DT = 0.005


def make_panels(gt, gen, steps, out_stem, n_curves=25, alpha=0.30, panel_fraction=0.24,
                panel_aspect=0.95):
    """gt, gen: [N, len(steps), 64]."""
    x = np.arange(gt.shape[-1]) * DX
    lo, hi = float(np.nanquantile(gt, 0.001)), float(np.nanquantile(gt, 0.999))
    pad = 0.08 * (hi - lo)
    lw = 0.9 * panel_fraction / 0.48

    use_paper_style()
    scale_panel_fonts(panel_fraction)
    for k, t in enumerate(steps):
        fig, ax = plt.subplots(figsize=figsize(panel_fraction, aspect=panel_aspect))
        curves = np.concatenate([gt[:n_curves, k], gen[:n_curves, k]], axis=0)
        cols = np.array([C_GT] * min(n_curves, len(gt)) + [C_BT] * min(n_curves, len(gen)))
        order = np.random.default_rng(0).permutation(len(curves))
        ax.add_collection(LineCollection([np.column_stack([x, curves[i]]) for i in order],
                                         colors=cols[order], linewidths=lw, alpha=alpha,
                                         zorder=2, rasterized=True))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel('$x$', labelpad=1)
        ax.set_ylabel('$u(x)$', labelpad=0)
        ax.tick_params(pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out_stem}_step{t}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f'wrote {out_stem}_step{t}.pdf  [tau={t * DT:g}]', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--enc-dir', required=True)
    p.add_argument('--test-data', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[100, 300, 500, 700])
    p.add_argument('--n-particles', type=int, default=1000)
    p.add_argument('--T-max', type=int, default=799)
    p.add_argument('--residual', action='store_true', help='must match training')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='paper_square')
    p.add_argument('--recompute', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, f'burgers_snapshots_{a.tag}.npz')
    if os.path.exists(cache) and not a.recompute:
        z = np.load(cache)
        gt, gen = z['gt'], z['gen']
    else:
        enc, dec, mean, std = load_autoencoder(a.enc_dir)
        model = load_generator(a.run_dir, T_max=a.T_max, residual=a.residual)
        raw = np.load(a.test_data, mmap_mode='r').astype(np.float32)
        idx = np.random.default_rng(a.seed).choice(raw.shape[0], min(a.n_particles, raw.shape[0]),
                                                   replace=False)
        T = max(a.steps)
        gt = np.array(raw[idx, :T + 1])
        gen = rollout(model, enc, dec, (gt[:, 0] - mean) / std, T, jax.random.PRNGKey(a.seed))
        gen = gen * std + mean
        gt, gen = gt[:, a.steps], gen[:, a.steps]
        np.savez_compressed(cache, gt=gt, gen=gen, steps=np.array(a.steps), dt=DT)
    make_panels(gt, gen, a.steps, os.path.join(a.out_dir, f'burgers_snapshots_{a.tag}'))


if __name__ == '__main__':
    main()
