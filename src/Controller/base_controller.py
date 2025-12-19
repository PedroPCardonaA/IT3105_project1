"""Abstract base interface for discrete-time controllers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp


class ControllerBase(ABC):
    """Common interface for controllers with reset, step, and simulate helpers."""

    dt: float
    dtype: jnp.dtype

    @abstractmethod
    def reset(self, state0: Any = None) -> Any:
        """Return initial controller state."""

    @abstractmethod
    def step(
        self,
        state: Any,
        y_ref: jnp.ndarray,
        y: jnp.ndarray,
        d: Union[jnp.ndarray, float] = 0.0,
    ) -> Tuple[Any, jnp.ndarray]:
        """Advance one step; return ``(next_state, control_action)``."""

    def simulate(
        self,
        y_ref_seq: jnp.ndarray,
        y_seq: jnp.ndarray,
        d_seq: jnp.ndarray,
        state0: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Scan over reference/measurement sequences using controller step dynamics."""

        def one_step(state, inputs):
            y_ref, y, d = inputs
            next_state, u = self.step(state, y_ref, y, d)
            return next_state, (next_state, u)

        _, (state_seq, u_seq) = jax.lax.scan(one_step, state0, (y_ref_seq, y_seq, d_seq))
        return state_seq, u_seq
