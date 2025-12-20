from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from .base_controller import ControllerBase

Array = jnp.ndarray


@dataclass(frozen=True)
class PIDState:
    """State container for PID controller."""
    e_init: Array
    e_prev: Array
    
    def tree_flatten(self):
        """Flatten for JAX PyTree."""
        return ((self.e_init, self.e_prev), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX PyTree."""
        return cls(*children)


# Register as JAX PyTree
register_pytree_node(PIDState, PIDState.tree_flatten, PIDState.tree_unflatten)


class PIDController(ControllerBase):
    """Classic PID controller implementing ``ControllerBase``."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        i_limit: Optional[float] = None,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        dtype=jnp.float32,
    ):
        self.theta = jnp.array([kp, ki, kd], dtype=dtype)
        self.dt = float(dt)
        self.i_limit = i_limit
        self.u_min = u_min
        self.u_max = u_max
        self.dtype = dtype

    @staticmethod
    def create_initial_state(dtype=jnp.float32) -> PIDState:
        """Create initial PID controller state."""
        zero = jnp.array(0.0, dtype=dtype)
        return PIDState(e_init=zero, e_prev=zero)

    @staticmethod
    def compute_error_terms(
        state: PIDState,
        e: Union[float, Array],
        dt: float,
        dtype=jnp.float32,
    ) -> Array:
        """Compute proportional, integral, and derivative error terms."""
        e = jnp.asarray(e, dtype=dtype)
        e = jnp.squeeze(e)

        dt_array = jnp.array(dt, dtype=dtype)

        # The integral is the accumulated error
        e_int = state.e_init + e * dt_array
        de_dt = (e - state.e_prev) / dt_array

        return jnp.array([e, e_int, de_dt], dtype=dtype)

    @staticmethod
    def compute_control(
        theta: Array,
        state: PIDState,
        e: Union[float, Array],
        dt: float,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        i_limit: Optional[float] = None,
        dtype=jnp.float32,
    ) -> Tuple[PIDState, Array]:
        """Compute PID control action and update state.

        Returns ``(next_state, control_output)`` with control wrapped to shape (1,).
        """
        theta = jnp.asarray(theta, dtype=dtype)
        kp, ki, kd = theta[0], theta[1], theta[2]

        e = jnp.asarray(e, dtype=dtype)
        e = jnp.squeeze(e)

        dt_array = jnp.array(dt, dtype=dtype)

        # Accumulate integral error
        e_int = state.e_init + e * dt_array

        if i_limit is not None:
            lim = jnp.array(float(i_limit), dtype=dtype)
            e_int = jnp.clip(e_int, -lim, lim)

        de_dt = (e - state.e_prev) / dt_array
        u = kp * e + ki * e_int + kd * de_dt

        if u_min is not None or u_max is not None:
            u = jnp.clip(u, a_min=u_min, a_max=u_max)

        next_state = PIDState(e_init=e_int, e_prev=e)
        return next_state, jnp.array([u], dtype=dtype)

    def reset(self, state0: Optional[PIDState] = None) -> PIDState:
        """Reset controller to initial state."""
        if state0 is None:
            return self.create_initial_state(dtype=self.dtype)
        return PIDState(
            e_init=jnp.array(state0.e_init, dtype=self.dtype),
            e_prev=jnp.array(state0.e_prev, dtype=self.dtype),
        )

    def step(
        self,
        state: PIDState,
        y_ref: jnp.ndarray,
        y: jnp.ndarray,
        d: Union[jnp.ndarray, float] = 0.0,
    ) -> Tuple[PIDState, jnp.ndarray]:
        """Compute one control step."""
        del d  # disturbance not used in classic PID
        e = jnp.asarray(y_ref - y, dtype=self.dtype)
        next_state, u = self.compute_control(
            self.theta,
            state,
            e,
            self.dt,
            u_min=self.u_min,
            u_max=self.u_max,
            i_limit=self.i_limit,
            dtype=self.dtype,
        )
        return next_state, u

    def get_error_terms(self, state: PIDState, y_ref: jnp.ndarray, y: jnp.ndarray) -> Array:
        """Get PID error terms [e, e_int, e_derivative]."""
        e = jnp.asarray(y_ref - y, dtype=self.dtype)
        return self.compute_error_terms(state, e, self.dt, dtype=self.dtype)