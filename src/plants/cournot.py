from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import jax.numpy as jnp
from .base import PlantBase
from ..utils import euler_step, rk4_step

@dataclass(frozen=True)
class CournotParams:
    """Market parameters for Cournot competition model."""
    pmax: float = 100.0 
    cm: float = 20.0

    #Quantity constraints
    qmin: float = 0.0
    qmax: float = 1.0

    #Clamp on price
    Umin: float = -0.05
    Umax: float = 0.05


def _profit_and_price(params: CournotParams, q1: jnp.ndarray, q2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute profit and market price based on quantities produced by two firms."""
    q = q1 + q2
    p = jnp.maximum(params.pmax - q, 0.0)
    P1 = q1 * (p - params.cm)
    return P1, p

def _deriv(params: CournotParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Compute derivative for controlled firm's quantity q1.
    
    The state is [q1, q2] but only q1 is controlled.
    q2 (competitor) changes due to disturbance d.
    """
    q1, q2 = state
    U = jnp.clip(u[0], params.Umin, params.Umax)
    D = d
    dq1 = U  # Firm 1 (controlled) rate of change
    dq2 = -D  # Firm 2 (competitor) affected by disturbance
    return jnp.array([dq1, dq2], dtype=state.dtype)

class CournotPlant(PlantBase):
    """Discrete-time Cournot competition plant with selectable integrator and outputs.

    Args:
        params (CournotParams): Market parameters.
        dt (float, optional): Integration step in seconds. Defaults to 1.0.
        integrator (Literal['rk4','euler'], optional): Numerical scheme. Defaults to 'rk4'.
        output (Literal['profit','price','full'], optional): Output view of the state. Defaults to 'profit'.
        dtype (jnp.dtype, optional): JAX dtype used internally. Defaults to jnp.float32.
    """
    def __init__(
            self,
            params: CournotParams,
            dt: float = 1.0,
            integrator: Literal['rk4', 'euler'] = 'rk4',
            output: Literal["profit", "price","q1", "q2", "full"] = "profit",
            dtype: jnp.dtype = jnp.float32,
    ):
        self.params = params
        self.dt = float(dt)
        self.integrator = integrator
        self.output_mode = output
        self.dtype = dtype

    def reset(self, state0: Tuple[float, float] = (0.0, 0.0)) -> jnp.ndarray:
        """Return the initial state array cast to the configured dtype."""
        q1, q2 = state0
        q1 = float(jnp.clip(q1, self.params.qmin, self.params.qmax))
        q2 = float(jnp.clip(q2, self.params.qmin, self.params.qmax))
        return jnp.array([q1, q2], dtype=self.dtype)
    
    def output(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select the output view (profit, price, quantities, or full state)."""
        q1, q2 = state
        P1, p = _profit_and_price(self.params, q1, q2)
        return {
            "profit": jnp.array([P1], dtype=state.dtype),
            "price":  jnp.array([p], dtype=state.dtype),
            "q1":     jnp.array([q1], dtype=state.dtype),
            "q2":     jnp.array([q2], dtype=state.dtype),
            "full":   jnp.array([q1, q2, p, P1], dtype=state.dtype),  # [q1,q2,price,profit]
        }[self.output_mode]
    
    def step(
            self,
            state: jnp.ndarray,
            u: jnp.ndarray,
            d: Union[jnp.ndarray, float] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Advance one discrete step and return `(next_state, output)`."""
        u = jnp.array(u, dtype=self.dtype)
        d = jnp.array(d, dtype=self.dtype)

        if self.integrator == "rk4":
            next_state = rk4_step(_deriv, self.params, state, u, d, self.dt)
        else:
            next_state = euler_step(_deriv, self.params, state, u, d, self.dt)

        next_state = jnp.clip(next_state, self.params.qmin, self.params.qmax)

        y = self.output(next_state)
        return next_state, y