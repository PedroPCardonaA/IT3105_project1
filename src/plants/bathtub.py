"""Discrete-time bathtub (tank) model utilities using JAX numerics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from .base import PlantBase
from ..utils import euler_step, rk4_step

@dataclass(frozen=True)
class BathtubParams:
    """Geometric and hydraulic parameters for the bathtub tank."""
    A: float = 1.0 
    C: float = 0.5
    g: float = 9.81

    Umin: float = 0.0
    Umax: float = 0.05

    Hmin: float = 0.0
    
    def tree_flatten(self):
        """Flatten for JAX PyTree."""
        values = (self.A, self.C, self.g, self.Umin, self.Umax, self.Hmin)
        keys = ('A', 'C', 'g', 'Umin', 'Umax', 'Hmin')
        return (values, keys)
    
    @classmethod
    def tree_unflatten(cls, keys, values):
        """Unflatten for JAX PyTree."""
        return cls(**dict(zip(keys, values)))


# Register as JAX PyTree
register_pytree_node(BathtubParams, BathtubParams.tree_flatten, BathtubParams.tree_unflatten)


def _outflow(patams: BathtubParams, H: jnp.ndarray) -> jnp.ndarray:
    """Compute outlet flow based on Torricelli's law with non-negative head."""
    H_pos = jnp.maximum(H, 0.0)
    V = jnp.sqrt(2 * patams.g * H_pos)
    Q = patams.C * V
    return Q

def _deriv(params: BathtubParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time head dynamics given inflow ``u`` and disturbance ``d``."""
    (H,) = state
    U = jnp.clip(u[0], params.Umin, params.Umax)
    Q = _outflow(params, H)
    dH = (U - Q + d) / params.A
    return jnp.array([dH], dtype=state.dtype)

class BathtubPlant(PlantBase):
    """Discrete-time bathtub plant with selectable integrator and outputs.

    Args:
        params (BathtubParams): Tank parameters.
        dt (float, optional): Integration step in seconds. Defaults to 1.0.
        integrator (Literal['rk4','euler'], optional): Numerical scheme. Defaults to 'rk4'.
        output (Literal['H','full'], optional): Output view of the state. Defaults to 'H'.
        dtype (jnp.dtype, optional): JAX dtype used internally. Defaults to jnp.float32.
    """
    def __init__(
            self,
            params: BathtubParams,
            dt: float = 1.0,
            integrator: Literal['rk4', 'euler'] = 'rk4',
            output: Literal["H", "full"] = "H",
            dtype: jnp.dtype = jnp.float32,
    ):
        self.params = params
        self.dt = float(dt)
        self.integrator = integrator
        self.output_mode = output
        self.dtype = dtype

    def reset(self, state0: Union[float, Tuple[float]]) -> jnp.ndarray:
        """Return the initial head state as a 1-element array."""

        if isinstance(state0, tuple):
            state0 = state0[0]
        state = jnp.array([state0], dtype=self.dtype)
        return state
    
    def output(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select head-only or full state output."""
        return {
            "H": state[0:1],
            "full": state,
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

        next_state = next_state.at[0].set(jnp.maximum(next_state[0], self.params.Hmin))
        
        y = self.output(next_state)
        return next_state, y