"""Debiased entropic-OT drift loss on feature sets (W-Flow, quadratic cost).

For one feature key: particles are rescaled by their mean pairwise distance
(per sqrt(dim)), the drift is V = T_pos - T_neg from Sinkhorn barycentric
projections onto the data and onto an independent model batch with
eps = R * dim, the conditioning block of V is zeroed (block-triangular
projection), V is normalized to unit RMS, and the loss regresses the model
features onto features + V.
"""

import jax
import jax.numpy as jnp


def _pairwise_sq_dist(X, Y):
    return jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)


def sinkhorn_log_plan(X, Y, eps, num_iter=20):
    """Log of the balanced entropic OT plan with cost 0.5 * |x - y|^2."""
    N, M = X.shape[0], Y.shape[0]
    logK = -0.5 * _pairwise_sq_dist(X, Y) / eps
    log_a = jnp.full((N,), -jnp.log(N))
    log_b = jnp.full((M,), -jnp.log(M))

    def body(_, uv):
        log_u, log_v = uv
        log_u = log_a - jax.nn.logsumexp(logK + log_v[None, :], axis=1)
        log_v = log_b - jax.nn.logsumexp(logK.T + log_u[None, :], axis=1)
        return log_u, log_v

    log_u, log_v = jax.lax.fori_loop(0, num_iter, body, (jnp.zeros((N,)), jnp.zeros((M,))))
    return log_u[:, None] + logK + log_v[None, :]


def barycentric_map(log_P, Y):
    return jnp.exp(log_P - jax.nn.logsumexp(log_P, axis=1, keepdims=True)) @ Y


def feature_scale(gen, pos, neg):
    old_gen = jax.lax.stop_gradient(gen)
    targets = jnp.concatenate([old_gen, jax.lax.stop_gradient(neg), jax.lax.stop_gradient(pos)])
    scale = jnp.sqrt(jnp.mean(_pairwise_sq_dist(old_gen, targets)))
    return jnp.clip(scale / jnp.sqrt(gen.shape[-1]), min=1e-3)


def drift_loss_single_key(gen, pos, neg, R, num_iter=20, d_anchor=0):
    """gen: [N, D] model features (differentiated); pos: [P, D] data features;
    neg: [Q, D] features of an independent model batch. The first d_anchor
    dimensions (the conditioning block) receive no drift."""
    D = gen.shape[-1]
    pos = jax.lax.stop_gradient(pos)
    neg = jax.lax.stop_gradient(neg)
    s = feature_scale(gen, pos, neg)
    old = jax.lax.stop_gradient(gen) / s
    pos_s, neg_s = pos / s, neg / s
    eps = R * float(D)
    V = (barycentric_map(sinkhorn_log_plan(old, pos_s, eps, num_iter), pos_s)
         - barycentric_map(sinkhorn_log_plan(old, neg_s, eps, num_iter), neg_s))
    if d_anchor > 0:
        V = V.at[:, :d_anchor].set(0.0)
    V = V / jnp.sqrt(jnp.clip(jnp.mean(V ** 2), min=1e-8))
    return jnp.mean((gen / s - (old + V)) ** 2)


def total_drift_loss(feats_gen, feats_pos, feats_neg, R, num_iter):
    """Sum over feature keys of the per-location-averaged loss on joint features."""
    losses = []
    for key in feats_gen:
        d_anchor = feats_gen[key].shape[-1] // 2
        per_loc = jax.vmap(lambda g, p, n: drift_loss_single_key(g, p, n, R, num_iter, d_anchor),
                           in_axes=(1, 1, 1))
        losses.append(jnp.mean(per_loc(feats_gen[key], feats_pos[key], feats_neg[key])))
    return jnp.sum(jnp.stack(losses))
