"""Stochastic Duffing oscillator data (Euler-Maruyama).

    dX1 = X2 dt
    dX2 = (-2 xi w X2 + w^2 X1 - w^2 gamma X1^3) dt + sigma dW

with xi = gamma = 0.2, w = 1, sigma = 0.5. Initial conditions are drawn from
N([0, -10], I_2) or fixed with --fixed-x0. Saves train.npy and test.npy of
shape [M, T+1, 2] to --out-dir.
"""

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from diffrax import (ODETerm, ControlTerm, MultiTerm, VirtualBrownianTree,
                     Euler, SaveAt, diffeqsolve)
import lineax

XI, GAMMA, W = 0.2, 0.2, 1.0


def duffing_drift(t, y, args=None):
    x1, x2 = y
    return jnp.array([x2, -2 * XI * W * x2 + W ** 2 * x1 - W ** 2 * GAMMA * x1 ** 3])


def make_duffing_diffusion(sigma):
    def duffing_diffusion(t, y, args=None):
        return jnp.array([0.0, sigma])
    return duffing_diffusion


def solve_sde_ic(y0, key, t_eval, dt, drift, diffusion):
    t0, tf = t_eval[0], t_eval[-1]
    bm = VirtualBrownianTree(t0, tf, tol=1e-3, shape=y0.shape, key=key)
    diag_diff = lambda *args: lineax.DiagonalLinearOperator(diffusion(*args))
    terms = MultiTerm(ODETerm(drift), ControlTerm(diag_diff, bm))
    sol = diffeqsolve(terms, Euler(), t0, tf, dt0=dt, y0=y0,
                      saveat=SaveAt(ts=t_eval), max_steps=int(2e6))
    return sol.ys


def generate_SDE_data(M, T, y0s, dt, key, drift, diffusion):
    """M trajectories of T steps from initial conditions y0s -> [M, T+1, 2]."""
    t_eval = jnp.linspace(0.0, T * dt, T + 1)

    @jit
    def solve_single(key, y0):
        return solve_sde_ic(y0, key, t_eval, dt, drift, diffusion)

    return vmap(solve_single)(random.split(key, M), y0s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--M-train', type=int, default=5000)
    p.add_argument('--M-test', type=int, default=5000)
    p.add_argument('--T', type=int, default=1200, help='number of time steps')
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0, help='seed for the initial conditions')
    p.add_argument('--noise-seed', type=int, default=None,
                   help='separate seed for the Brownian paths (default: derived from --seed)')
    p.add_argument('--fixed-x0', type=float, nargs=2, default=None, metavar=('X1', 'X2'),
                   help='start every trajectory from this initial condition')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'JAX devices: {jax.devices()}', flush=True)

    M_total = args.M_train + args.M_test
    key = random.PRNGKey(args.seed)
    key, k_ic = random.split(key)
    if args.fixed_x0 is not None:
        y0s = jnp.tile(jnp.array(args.fixed_x0), (M_total, 1))
    else:
        y0s = jnp.array([0.0, -10.0]) + random.normal(k_ic, shape=(M_total, 2))

    key, _, _ = random.split(key, 3)
    noise_key = key if args.noise_seed is None else random.PRNGKey(args.noise_seed)
    print(f'Generating {M_total} trajectories, T={args.T}, dt={args.dt}, sigma={args.sigma}',
          flush=True)
    data = np.array(generate_SDE_data(M_total, args.T, y0s, args.dt, noise_key,
                                      duffing_drift, make_duffing_diffusion(args.sigma)))

    np.save(os.path.join(args.out_dir, 'train.npy'), data[:args.M_train])
    np.save(os.path.join(args.out_dir, 'test.npy'), data[args.M_train:])
    print(f'train {data[:args.M_train].shape}, test {data[args.M_train:].shape} -> {args.out_dir}',
          flush=True)


if __name__ == '__main__':
    main()
