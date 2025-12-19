"""Shared numerical integration utilities for plant models."""

from __future__ import annotations
from typing import Callable, Any

import jax.numpy as jnp

DerivFn = Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def rk4_step(deriv_fn: DerivFn, params: Any, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Advance state one step with classic fourth-order Runge–Kutta."""
    k1 = deriv_fn(params, state, u, d)
    k2 = deriv_fn(params, state + 0.5 * dt * k1, u, d)
    k3 = deriv_fn(params, state + 0.5 * dt * k2, u, d)
    k4 = deriv_fn(params, state + dt * k3, u, d)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(deriv_fn: DerivFn, params: Any, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Advance state one step with forward Euler integration."""
    return state + dt * deriv_fn(params, state, u, d)
