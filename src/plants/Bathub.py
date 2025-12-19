from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import jax.numpy as jnp

from .base import PlantBase
from ..utils import euler_step, rk4_step

@dataclass(frozen=True)
class BathubParams:
    A: float = 1.0 
    C: float = 0.5
    g: float = 9.81

    Umin: float = 0.0
    Umax: float = 0.05

    Hmin: float = 0.0


def _outflow(patams: BathubParams, H: jnp.ndarray) -> jnp.ndarray:
    H_pos = jnp.maximum(H, 0.0)
    V = jnp.sqrt(2 * patams.g * H_pos)
    Q = patams.C * V
    return Q

def _deriv(params: BathubParams, state: jnp.ndarray, u: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    (H,) = state
    U = jnp.clip(u[0], params.Umin, params.Umax)
    Q = _outflow(params, H)
    dH = (U - Q + d) / params.A
    return jnp.array([dH], dtype=state.dtype)

class BathubPlant(PlantBase):
    def __init__(
            self,
            params: BathubParams,
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

        if isinstance(state0, tuple):
            state0 = state0[0]
        state = jnp.array(state0, dtype=self.dtype)
        return state
    
    def output(self, state: jnp.ndarray) -> jnp.ndarray:
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
        u = jnp.array(u, dtype=self.dtype)
        d = jnp.array(d, dtype=self.dtype)

        if self.integrator == "rk4":
            next_state = rk4_step(_deriv, self.params, state, u, d, self.dt)
        else:
            next_state = euler_step(_deriv, self.params, state, u, d, self.dt)

        next_state = next_state.at[0].set(jnp.maximum(next_state[0], self.params.Hmin))
        
        y = self.output(next_state)
        return next_state, y