from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True)
class PIDState:
    """State container for PID controller."""
    e_init: Array
    e_prev: Array

def pid_reset(dtype = jnp.float32) -> PIDState:
    """Reset PID controller state."""
    zero = jnp.array(0.0, dtype=dtype)
    return PIDState(e_init=zero, e_prev=zero)

def pid_step(
        theta: Array,
        state: PIDState,
        e: Union[float, Array],
        dt: float,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        i_limit: Optional[float] = None,
        dtype = jnp.float32,
) -> Tuple[Array, PIDState]:
    """Compute PID control action and update state.
    Args:
        theta (Array): PID gains [kp, ki, kd].
        state (PIDState): Current PID controller state.
        e (Union[float, Array]): Current error signal.
        dt (float): Time step duration.
        u_min (Optional[float], optional): Minimum control output. Defaults to None.
        u_max (Optional[float], optional): Maximum control output. Defaults to None.
        i_limit (Optional[float], optional): Integral windup limit. Defaults to None.
        dtype (_type_, optional): JAX dtype for computations. Defaults to jnp.float32.
    Returns:
        Tuple[Array, PIDState]: Control output and updated PID state.
    """
    theta = jnp.asarray(theta, dtype=dtype)
    kp, ki, kd = theta[0], theta[1], theta[2]


    e = jnp.asarray(e, dtype=dtype)
    e = jnp.squeeze(e)

    dt_array = jnp.array(dt, dtype=dtype)

    e_int = state.e_init * dt_array

    if i_limit is not None:
        lim = jnp.array(float(i_limit), dtype=dtype)
        e_int = jnp.clip(e_int, -lim, lim)
    
    de_dt = (e - state.e_prev) / dt_array
    u = kp * e + ki * e_int + kd * de_dt

    if u_min is not None or u_max is not None:
        u = jnp.clip(u, a_min=u_min, a_max=u_max)

    next_state = PIDState(e_init=e_int / dt_array, e_prev=e)
    return jnp.array([u], dtype=dtype), next_state
        


def pid_error_terms(state: PIDState, e: Union[float, Array], dt: float, dtype = jnp.float32) -> Array:
    """Compute proportional, integral, and derivative error terms for PID controller."""
    e = jnp.asarray(e, dtype=dtype)
    e = jnp.squeeze(e)

    dt_array = jnp.array(dt, dtype=dtype)

    e_int = state.e_init * dt_array
    de_dt = (e - state.e_prev) / dt_array

    return jnp.array([e, e_int, de_dt], dtype=dtype)