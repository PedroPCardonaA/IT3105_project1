"""Abstract base interface for discrete-time plant models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax
import jax.numpy as jnp


class PlantBase(ABC):
    """Common interface for plants with reset, output, step, and simulate.

    Implementations should define how to initialize state, map state to outputs,
    and advance dynamics one discrete step. A shared ``simulate`` convenience is
    provided using ``jax.lax.scan``.
    """

    dt: float
    dtype: jnp.dtype

    @abstractmethod
    def reset(self, state0: Any) -> jnp.ndarray:
        """Return initial state for the plant as a JAX array."""

    @abstractmethod
    def output(self, state: jnp.ndarray) -> jnp.ndarray:
        """Return plant-specific outputs from state."""

    @abstractmethod
    def step(self, state: jnp.ndarray, u: jnp.ndarray, d=0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Advance plant by one step and return next state and outputs."""

    def simulate(self, u_seq: jnp.ndarray, d_seq: jnp.ndarray, state0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Scan over sequences of inputs/disturbances using plant step dynamics."""

        def one_step(state, inputs):
            u, d = inputs
            next_state, y = self.step(state, u, d)
            return next_state, (next_state, y)

        _, (state_seq, y_seq) = jax.lax.scan(one_step, state0, (u_seq, d_seq))
        return state_seq, y_seq
