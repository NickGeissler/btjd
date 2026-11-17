"""Turbulence vorticity snapshots, ground truth vs BTJD (Figure: turbulence trajectories).

Rolls out --n-traj test initial conditions and writes, for each of the first
--show-traj of them, a two-row panel (ground truth over BTJD) at the given
rollout steps (cfd_snapshots_ic{i}), plus one combined figure. Colour scale is
symmetric and shared per column. Fields are cached to <out-dir>/cfd_snapshots.npz;
--from-npz re-renders from the cache without a rollout.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax

from common.plot_style import use_paper_style, TEXTWIDTH_IN, TICK_PT
from cfd.autoencoder import load_autoencoder
from cfd.evaluate import rollout_latent, decode
from cfd.generator import load_generator

DT = 0.1001                 # physical time per rollout step
LABEL_PT = TICK_PT - 1
CMAP = 'berlin'             # zero vorticity renders black
MARGIN = dict(left=0.62, right=0.42, bottom=0.46, top=0.06)


def draw_rules(fig, axes, row_lw, color='white'):
    """Frame, column dividers, and row dividers of width row_lw(row_index)."""
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    y0, y1 = axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1

    def line(xs, ys, lw):
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=color, lw=lw, zorder=5))

    line([x0, x1], [y0, y0], 1.1); line([x0, x1], [y1, y1], 1.1)
    line([x0, x0], [y0, y1], 1.1); line([x1, x1], [y0, y1], 1.1)
    for c in range(axes.shape[1] - 1):
        x = 0.5 * (axes[0, c].get_position().x1 + axes[0, c + 1].get_position().x0)
        line([x, x], [y0, y1], 1.1)
    for r in range(1, axes.shape[0]):
        y = 0.5 * (axes[r - 1, 0].get_position().y0 + axes[r, 0].get_position().y1)
        line([x0, x1], [y, y], row_lw(r))


def snapshot_figure(gt, gen, steps, out_stem, row_traj):
    """gt, gen: [n_ic, n_steps, H, W]."""
    use_paper_style()
    n_ic, n_cols = gt.shape[0], len(steps)
    right = MARGIN['right'] if n_ic > 1 else 0.08
    panel = (TEXTWIDTH_IN - MARGIN['left'] - right) / n_cols
    fig_h = 2 * n_ic * panel + MARGIN['bottom'] + MARGIN['top']
    fig = plt.figure(figsize=(TEXTWIDTH_IN, fig_h))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2 * n_ic, n_cols, wspace=0.0, hspace=0.0,
                          left=MARGIN['left'] / TEXTWIDTH_IN, right=1 - right / TEXTWIDTH_IN,
                          bottom=MARGIN['bottom'] / fig_h, top=1 - MARGIN['top'] / fig_h)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(2 * n_ic)])
    for c, step in enumerate(steps):
        vmax = float(max(np.abs(gt[:, c]).max(), np.abs(gen[:, c]).max()))
        for r in range(n_ic):
            for k, field in enumerate((gt[r, c], gen[r, c])):
                ax = axes[2 * r + k, c]
                ax.imshow(field, origin='lower', cmap=CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor('white')
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c == 0:
                    ax.set_ylabel('Physics Model' if k == 0 else 'BTJD (ours)',
                                  fontsize=LABEL_PT, color='black', labelpad=5)
        axes[-1, c].set_xlabel(f'step {step}\n$\\tau = {step * DT:.1f}$',
                               fontsize=LABEL_PT, color='black', labelpad=5)
    if n_ic > 1:
        for r in range(n_ic):
            top, bot = axes[2 * r, -1].get_position(), axes[2 * r + 1, -1].get_position()
            fig.text(top.x1 + 0.012, 0.5 * (top.y1 + bot.y0), f'IC {row_traj[r]}', rotation=270,
                     va='center', ha='left', fontsize=LABEL_PT, color='black')
    draw_rules(fig, axes, lambda r: 2.0 if r % 2 == 0 else 0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', dpi=400, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf', flush=True)


def render(gt, gen, steps, out_dir):
    for i in range(gt.shape[0]):
        snapshot_figure(gt[i:i + 1], gen[i:i + 1], steps, os.path.join(out_dir, f'cfd_snapshots_ic{i}'),
                        [i])
    snapshot_figure(gt, gen, steps, os.path.join(out_dir, 'cfd_snapshots'), list(range(gt.shape[0])))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt')
    p.add_argument('--ae-dir')
    p.add_argument('--latent-data')
    p.add_argument('--pixel-data')
    p.add_argument('--data-scale', type=float)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[25, 50, 75, 100, 125, 150])
    p.add_argument('--n-traj', type=int, default=8)
    p.add_argument('--show-traj', type=int, default=3)
    p.add_argument('--time-stride', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--from-npz', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, 'cfd_snapshots.npz')
    if a.from_npz:
        d = np.load(cache)
        render(d['gt'], d['gen'], list(d['steps']), a.out_dir)
        return
    s, T = a.time_stride, max(a.steps)
    lat = np.load(a.latent_data, mmap_mode='r')
    pix = np.load(a.pixel_data, mmap_mode='r')
    n = min(a.n_traj, lat.shape[0])
    gt = np.array(pix[:n, :T * s + 1:s]) * a.data_scale
    model, ae = load_generator(a.ckpt), load_autoencoder(a.ae_dir)
    gen = decode(ae, rollout_latent(model, np.array(lat[:n, 0]), T, jax.random.PRNGKey(a.seed), s))
    gen = gen * a.data_scale
    k = min(a.show_traj, n)
    gt_sel = np.stack([gt[:k, t, :, :, 0] for t in a.steps], axis=1)
    gen_sel = np.stack([gen[:k, t, :, :, 0] for t in a.steps], axis=1)
    np.savez(cache, steps=np.array(a.steps), gt=gt_sel, gen=gen_sel, dt=DT, time_stride=s,
             data_scale=a.data_scale)
    render(gt_sel, gen_sel, a.steps, a.out_dir)


if __name__ == '__main__':
    main()
