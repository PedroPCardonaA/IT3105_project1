"""Discrete-time DC motor model utilities using JAX numerics."""

from __future__ import annotations
from dataclasses import dataclass 
from typing import Literal, Tuple  

import jax
import jax.numpy as jnp

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
    i, omega, theta = state

    V = jnp.clip(u[0], -params.Vmax, params.Vmax)
    di = (-params.R * i - params.Ke * omega + V) / params.L
    tau_f = params.tau_c * jnp.tanh(omega / params.omega_s)
    tau_load = params.tau1 * jnp.sin(theta)
    domega = (params.Kt * i - params.b * omega - tau_f - tau_load - d)/ params.J
    dtheta = omega

    return jnp.array([di, domega, dtheta], dtype=state.dtype)

def _rk4_step(params: DCMotorParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Advance the state by one Runge–Kutta 4 integration step."""
    k1 = _deriv(params, state, u, d)
    k2 = _deriv(params, state + 0.5 * dt * k1, u, d)
    k3 = _deriv(params, state + 0.5 * dt * k2, u, d)
    k4 = _deriv(params, state + dt * k3, u, d)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def _euler_step(params: DCMotorParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Advance the state by one forward Euler integration step."""
    return state + dt * _deriv(params, state, u, d)


class DCMotorPlant:
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
    
    def step(self, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray = 0.0) -> jnp.ndarray:
        """Advance one discrete step and return `(next_state, output)`."""
        u = jnp.array(u, dtype=self.dtype)
        d = jnp.array(d, dtype=self.dtype)

        if self.integrator == 'rk4':
            next_state = _rk4_step(self.params, state, u, d, self.dt)
        else:
            next_state = _euler_step(self.params, state, u, d, self.dt)
        y = self.output(next_state)
        return next_state, y
    

    def simulate(plant: DCMotorPlant, u_seq: jnp.ndarray, d_seq: jnp.ndarray, state0: jnp.ndarray):
        """Simulate a DC motor over sequences of inputs and disturbances."""
        def one_step(state, inputs):
            u, d = inputs
            next_state, y = plant.step(state, u, d)
            return next_state, (next_state, y)
    
        _, (state_seq, y_seq) = jax.lax.scan(one_step, state0, (u_seq, d_seq))
        return state_seq, y_seq