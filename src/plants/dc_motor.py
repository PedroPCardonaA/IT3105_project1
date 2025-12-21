"""Discrete-time DC motor model utilities using JAX numerics."""

from __future__ import annotations
from dataclasses import dataclass 
from typing import Literal, Tuple, Union  

import jax.numpy as jnp

from .base import PlantBase
from ..utils import euler_step, rk4_step

@dataclass(frozen=True)
class DCMotorParams:
    """Electrical, mechanical, and friction parameters for a DC motor."""

    # Electrical parameters
    R: float = 2.0
    L: float = 0.5
    Ke: float = 0.1

    # Mechanical parameters
    Kt: float = 0.1
    J: float = 0.02
    b: float = 0.001

    # Nonlinear friction
    tau_c: float = 0.02
    omega_s: float = 0.1

    # position-dependt load
    tau1: float = 0.05

    # Saturation
    Vmax: float = 12.0

def _deriv(params: DCMotorParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Compute continuous-time state derivatives for the DC motor dynamics."""
    i, omega, theta = state[0], state[1], state[2]

    V = jnp.clip(u[0], -params.Vmax, params.Vmax)
    
    di = (-params.R * i - params.Ke * omega + V) / params.L
    tau_f = params.tau_c * jnp.tanh(omega / params.omega_s)
    tau_load = params.tau1 * jnp.sin(theta)
    domega = (params.Kt * i - params.b * omega - tau_f - tau_load - d) / params.J
    dtheta = omega

    return jnp.array([di, domega, dtheta], dtype=state.dtype)

class DCMotorPlant(PlantBase):
    """Represents a discrete-time DC motor plant simulation with configurable integration and outputs.
    Args:
        params (DCMotorParams, optional): Motor parameters instance.
        dt (float, optional): Discrete-time step for integration (seconds). Defaults to 0.01.
        integrator (Literal['rk4', 'euler'], optional): Numerical integration method. Defaults to 'rk4'.
        output (Literal['omega', 'theta', 'full'], optional): Desired output view of the state. Defaults to 'full'.
        dtype (jnp.dtype, optional): JAX dtype for internal arrays. Defaults to jnp.float32.
    Methods:
        reset(state0: Tuple[float, float, float]) -> jnp.ndarray:
            Initialize and return the motor state from a given tuple (current, angular velocity, angular position).
        output(state: jnp.ndarray) -> jnp.ndarray:
            Return the selected output slice of the state based on `output` setting.
        step(state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Advance the system one time step using the chosen integrator with control input `u` and disturbance `d`.
    """

    def __init__(
            self,
            params: DCMotorParams = DCMotorParams(),
            dt: float = 0.01,
            integrator: Literal['rk4', 'euler'] = 'rk4',
            output: Literal['omega', 'theta', 'full'] = 'full',
            dtype: jnp.dtype = jnp.float32,
    ):
        self.params = params
        self.dt = float(dt)
        self.integrator = integrator
        self.output_kind = output
        self.dtype = dtype

    def reset(self, state0: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> jnp.ndarray:
        """Return the initial state array cast to the configured dtype."""
        state = jnp.array(state0, dtype=self.dtype)
        return state
    
    def output(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select the output view (omega, theta, or full state)."""
        return {
            'omega': state[1:2],
            'theta': state[2:3],
            'full': state,
        }[self.output_kind]
    
    def step(self, state: jnp.ndarray, u: jnp.ndarray, d: Union[jnp.ndarray, float] = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Advance one discrete step and return `(next_state, output)`."""
        u = jnp.array(u, dtype=self.dtype)
        d = jnp.array(d, dtype=self.dtype)

        if self.integrator == 'rk4':
            next_state = rk4_step(_deriv, self.params, state, u, d, self.dt)
        else:
            next_state = euler_step(_deriv, self.params, state, u, d, self.dt)
        y = self.output(next_state)
        return next_state, y