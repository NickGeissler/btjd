"""Duffing trajectories from one shared initial condition (Figure: DuffingSameIC).

Draws --n-env ground-truth SDE realizations and --n-env BTJD rollouts from the
same initial condition, and writes one panel per arm (_gt, _liftdrift): a
log-scaled occupancy density of the full ensemble with --n-show individual
paths on top. Both panels share the density scale and axis limits.

The paper figure uses the initial condition of test trajectory 0 and --seed 7.
Rollouts are cached to <out-dir>/ensemble_<tag>.npz.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import jax
import jax.numpy as jnp

from common.plot_style import C_GT, C_BT, use_paper_style, figsize, scale_panel_fonts
from duffing.generate_data import duffing_drift, make_duffing_diffusion, generate_SDE_data
from duffing.model import load_model

WELL = float(np.sqrt(1.0 / 0.2))


def gt_ensemble(x0, n, T, dt, sigma, seed):
    y0s = jnp.tile(jnp.asarray(x0, dtype=jnp.float32), (n, 1))
    return np.asarray(generate_SDE_data(n, T, y0s, dt, jax.random.PRNGKey(seed),
                                        duffing_drift, make_duffing_diffusion(sigma)))


def model_ensemble(model, mean, std, x0, n, T, seed):
    step = jax.jit(jax.vmap(model.step, in_axes=(0, 0, None)))
    x = jnp.tile(jnp.asarray((np.asarray(x0, dtype=np.float32) - mean) / std), (n, 1))
    key = jax.random.PRNGKey(seed)
    traj = [np.array(x)]
    for t in range(T):
        key, k = jax.random.split(key)
        x = step(x, jax.random.normal(k, x.shape), float(t))
        traj.append(np.array(x))
    return np.stack(traj, axis=1) * std + mean


def make_panels(gt, bt, n_show, out_stem, xlim, ylim, div_thresh=20.0, bins=260,
                panel_fraction=0.48, panel_aspect=0.85):
    ok = lambda d: np.isfinite(d).all(axis=(1, 2)) & (np.abs(d).max(axis=(1, 2)) < div_thresh)
    arms = [('gt', gt[ok(gt)], C_GT), ('liftdrift', bt[ok(bt)], C_BT)]
    hist = {}
    for name, d, _ in arms:
        pts = d.reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        hist[name] = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[xlim, ylim])
    vmax = max(h.max() for h, _, _ in hist.values())
    x0 = gt[0, 0]

    use_paper_style()
    scale_panel_fonts(panel_fraction)
    for name, d, color in arms:
        fig, ax = plt.subplots(figsize=figsize(panel_fraction, aspect=panel_aspect))
        h, xe, ye = hist[name]
        ax.pcolormesh(xe, ye, h.T, cmap=LinearSegmentedColormap.from_list('fade', ['#ffffff', color]),
                      norm=LogNorm(vmin=1, vmax=vmax), rasterized=True, zorder=1)
        for i in range(min(n_show, len(d))):
            ax.plot(d[i, :, 0], d[i, :, 1], lw=0.3, alpha=0.7, color=color, zorder=2)
        ax.plot(x0[0], x0[1], marker='o', ms=3.6, color='#000000', zorder=5)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out_stem}_{name}.{ext}', dpi=500, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f'wrote {out_stem}_{name}.pdf', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--test-data', default=None, help='take the IC from this set')
    p.add_argument('--ic-traj', type=int, default=0)
    p.add_argument('--x0', type=float, nargs=2, default=None, help='explicit IC instead')
    p.add_argument('--T', type=int, default=1200)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('--n-env', type=int, default=2048)
    p.add_argument('--n-show', type=int, default=20)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--time-enc', default='scalar', choices=['scalar', 'sinusoidal'])
    p.add_argument('--T-max', type=int, default=1200)
    p.add_argument('--seed', type=int, default=7,
                   help='GT noise seed; the model rollout uses seed+1')
    p.add_argument('--tag', default='paper')
    p.add_argument('--recompute', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, f'ensemble_{a.tag}.npz')
    if os.path.exists(cache) and not a.recompute:
        z = np.load(cache)
        gt, bt = z['gt'], z['bt']
    else:
        if a.x0 is not None:
            x0 = np.array(a.x0, dtype=np.float32)
        else:
            x0 = np.array(np.load(a.test_data, mmap_mode='r')[a.ic_traj, 0], dtype=np.float32)
        print(f'shared IC: {x0}', flush=True)
        gt = gt_ensemble(x0, a.n_env, a.T, a.dt, a.sigma, a.seed)
        model, mean, std = load_model(a.run_dir, a.hidden, a.n_layers, a.T_max,
                                      time_enc=a.time_enc)
        bt = model_ensemble(model, mean, std, x0, a.n_env, a.T, a.seed + 1)
        np.savez_compressed(cache, gt=gt, bt=bt, x0=x0, T=a.T)

    lim = 1.05 * max(np.abs(np.nanquantile(gt[..., 0], [0.001, 0.999])).max(), WELL)
    vlim = 1.05 * np.abs(np.nanquantile(gt[..., 1], [0.001, 0.999])).max()
    make_panels(gt, bt, a.n_show, os.path.join(a.out_dir, f'paper_ensemble_{a.tag}'),
                xlim=(-lim, lim), ylim=(-vlim, vlim))


if __name__ == '__main__':
    main()
