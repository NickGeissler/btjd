"""Log-domain Sinkhorn barycentric projection used by the low-dimensional
(Duffing, Rayleigh-Benard) and Burgers drift fields."""

import jax
import jax.numpy as jnp


def sinkhorn_projection(gen_particles, target_particles, eps, iters, w_xt=1.0):
    """Barycentric projection of each generated particle onto the target cloud.

    gen_particles, target_particles: [n, 2*d] joint samples (x_t, x_{t+1}).
    w_xt rescales the x_t block of the squared-Euclidean cost.
    """
    a = jnp.ones(gen_particles.shape[0]) / gen_particles.shape[0]
    b = jnp.ones(target_particles.shape[0]) / target_particles.shape[0]
    scale = jnp.array([w_xt] * (gen_particles.shape[-1] // 2)
                      + [1.0] * (gen_particles.shape[-1] // 2))
    gen_scaled = gen_particles * scale
    target_scaled = target_particles * scale
    C = jnp.sum((gen_scaled[:, None] - target_scaled[None, :]) ** 2, axis=-1)
    log_K = -C / eps
    log_v = jnp.zeros_like(b)
    for _ in range(iters):
        log_u = jnp.log(a) - jax.nn.logsumexp(log_K + log_v[None, :], axis=1)
        log_v = jnp.log(b) - jax.nn.logsumexp(log_K + log_u[:, None], axis=0)
    log_w = log_u[:, None] + log_K + log_v[None, :]
    log_w = log_w - jax.nn.logsumexp(log_w, axis=1, keepdims=True)
    return jnp.exp(log_w) @ target_particles
