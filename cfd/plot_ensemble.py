"""Same-IC turbulence ensemble (Figure: turbulence ensemble).

One ground-truth row and --n-rows independent BTJD rollouts from the same
latent initial condition. With --op-rollouts (the operator-learning baseline's
rollouts from the same IC, an npz with 'runs' of shape [n, 1000, 64, 64, 1] on
the solver's stride-1 grid) its rows are appended, decimated to the same
stride-4 grid. Fields are cached to <out-dir>/cfd_ensemble.npz; --from-npz
re-renders from the cache.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax

from common.plot_style import use_paper_style, TEXTWIDTH_IN
from cfd.autoencoder import load_autoencoder
from cfd.evaluate import rollout_latent, decode
from cfd.generator import load_generator
from cfd.plot_snapshots import CMAP, DT, LABEL_PT, draw_rules

MARGIN = dict(left=0.92, right=0.06, bottom=0.42, top=0.05)


def row_labels(n_ours, n_op):
    return (['Physics Model'] + [f'BTJD (ours)\nsample {i + 1}' for i in range(n_ours)]
            + [f'Operator\nLearning\nsample {i + 1}' for i in range(n_op)])


def ensemble_figure(fields, labels, steps, out_stem):
    """fields: [n_rows, n_steps, H, W]; row 0 is ground truth."""
    use_paper_style()
    n_rows, n_cols = fields.shape[0], len(steps)
    panel = (TEXTWIDTH_IN - MARGIN['left'] - MARGIN['right']) / n_cols
    fig_h = n_rows * panel + MARGIN['bottom'] + MARGIN['top']
    fig = plt.figure(figsize=(TEXTWIDTH_IN, fig_h))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.0, hspace=0.0,
                          left=MARGIN['left'] / TEXTWIDTH_IN,
                          right=1 - MARGIN['right'] / TEXTWIDTH_IN,
                          bottom=MARGIN['bottom'] / fig_h, top=1 - MARGIN['top'] / fig_h)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)])
    for c, step in enumerate(steps):
        vmax = float(np.abs(fields[:, c]).max())
        for r in range(n_rows):
            ax = axes[r, c]
            ax.imshow(fields[r, c], origin='lower', cmap=CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=LABEL_PT, color='black', rotation=0, ha='right',
                              va='center', labelpad=5, linespacing=1.25)
        label = 'step 0\nfixed IC' if step == 0 else f'step {step}\n$\\tau = {step * DT:.1f}$'
        axes[-1, c].set_xlabel(label, fontsize=LABEL_PT, color='black', labelpad=5)
    draw_rules(fig, axes, lambda r: 2.0 if r == 1 else 1.1)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', dpi=400, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt')
    p.add_argument('--ae-dir')
    p.add_argument('--latent-data')
    p.add_argument('--pixel-data')
    p.add_argument('--data-scale', type=float)
    p.add_argument('--op-rollouts', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--out-stem', default=None,
                   help='default: cfd_ensemble_snapshots[_with_op]')
    p.add_argument('--steps', type=int, nargs='+', default=[0, 10, 20, 30, 40, 50])
    p.add_argument('--n-rows', type=int, default=3)
    p.add_argument('--ic-index', type=int, default=0)
    p.add_argument('--time-stride', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--from-npz', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, 'cfd_ensemble.npz')
    stem = lambda n_op: os.path.join(a.out_dir, a.out_stem or
                                     ('cfd_ensemble_snapshots_with_op' if n_op else 'cfd_ensemble_snapshots'))
    if a.from_npz:
        d = np.load(cache)
        ensemble_figure(d['fields'], row_labels(int(d['n_ours']), int(d['n_op'])), list(d['steps']),
                        stem(int(d['n_op'])))
        return
    s = a.time_stride
    z0 = np.repeat(np.array(np.load(a.latent_data, mmap_mode='r')[a.ic_index, 0])[None],
                   a.n_rows, axis=0)
    model, ae = load_generator(a.ckpt), load_autoencoder(a.ae_dir)
    gen = decode(ae, rollout_latent(model, z0, max(a.steps), jax.random.PRNGKey(a.seed), s))
    gen = np.stack([gen[:, t, :, :, 0] for t in a.steps], axis=1) * a.data_scale
    pix = np.load(a.pixel_data, mmap_mode='r')
    gt = np.stack([np.array(pix[a.ic_index, t * s, :, :, 0]) for t in a.steps]) * a.data_scale
    rows = [gt[None], gen]
    n_op = 0
    if a.op_rollouts:
        op = np.load(a.op_rollouts)['runs'][:, ::s]
        rows.append(np.stack([op[:, t, :, :, 0] for t in a.steps], axis=1))
        n_op = op.shape[0]
    fields = np.concatenate(rows, axis=0)
    np.savez(cache, fields=fields, steps=np.array(a.steps), n_ours=a.n_rows, n_op=n_op,
             ic_index=a.ic_index, dt=DT, time_stride=s, data_scale=a.data_scale)
    ensemble_figure(fields, row_labels(a.n_rows, n_op), a.steps, stem(n_op))


if __name__ == '__main__':
    main()
