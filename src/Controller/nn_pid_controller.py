from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from .base_controller import ControllerBase

Array = jnp.ndarray


@dataclass(frozen=True)
class NN_PIDState:
    """State container for NN-PID controller."""
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
register_pytree_node(NN_PIDState, NN_PIDState.tree_flatten, NN_PIDState.tree_unflatten)


class NNPIDController(ControllerBase):
    """Neural-network-based PID controller implementing ``ControllerBase``."""

    def __init__(
        self,
        layer_sizes: Tuple[int, ...] = (3, 32, 1),
        hidden_activation: Literal["sigmoid", "tanh", "relu"] = "tanh",
        dt: float = 0.01,
        w_init_range: Tuple[float, float] = (-0.1, 0.1),
        b_init_range: Tuple[float, float] = (-0.1, 0.1),
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        i_limit: Optional[float] = None,
        dtype=jnp.float32,
        key: Array = jax.random.PRNGKey(0),
    ):
        self.params = self._initialize_mlp_params(
            key, layer_sizes, w_init_range, b_init_range, dtype=dtype
        )
        self.hidden_activation: Literal["sigmoid", "tanh", "relu"] = hidden_activation
        self.dt = float(dt)
        self.u_min = u_min
        self.u_max = u_max
        self.i_limit = i_limit
        self.dtype = dtype

    @staticmethod
    def create_initial_state(dtype=jnp.float32) -> NN_PIDState:
        """Create initial NN-PID controller state."""
        z = jnp.array(0.0, dtype=dtype)
        return NN_PIDState(e_init=z, e_prev=z)

    @staticmethod
    def get_activation_fn(
        name: Literal["sigmoid", "tanh", "relu"]
    ) -> Callable[[Array], Array]:
        """Get activation function by name."""
        if name == "sigmoid":
            return jnn.sigmoid
        if name == "tanh":
            return jnp.tanh
        if name == "relu":
            return jnn.relu
        raise ValueError(f"Unknown activation: {name}")

    @staticmethod
    def _initialize_mlp_params(
        key: Array,
        layer_sizes: Tuple[int, ...],
        w_init_range: Tuple[float, float] = (-0.1, 0.1),
        b_init_range: Tuple[float, float] = (-0.1, 0.1),
        dtype=jnp.float32,
    ) -> List:
        """Initialize MLP parameters with uniform distribution."""
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        lo_w, hi_w = w_init_range
        lo_b, hi_b = b_init_range

        params = []
        layer_keys = jax.random.split(key, len(layer_sizes) - 1)
        for k, (din, dout) in zip(layer_keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            w_key, b_key = jax.random.split(k)
            W = jax.random.uniform(w_key, (din, dout), dtype=dtype, minval=lo_w, maxval=hi_w)
            b = jax.random.uniform(b_key, (dout,), dtype=dtype, minval=lo_b, maxval=hi_b)
            params.append({"W": W, "b": b})
        return params

    @staticmethod
    def _mlp_forward(
        params,
        x: Array,
        hidden_activation: Literal["sigmoid", "tanh", "relu"] = "tanh",
    ) -> Array:
        """Forward pass returning a scalar output."""
        h = x
        f = NNPIDController.get_activation_fn(hidden_activation)
        for layer in params[:-1]:
            h = f(h @ layer["W"] + layer["b"])
        last = params[-1]
        y = h @ last["W"] + last["b"]
        return jnp.squeeze(y)

    @staticmethod
    def compute_control(
        nn_params,
        state: NN_PIDState,
        e: Union[float, Array],
        dt: float,
        hidden_activation: Literal["sigmoid", "tanh", "relu"] = "tanh",
        u_min: Union[float, None] = None,
        u_max: Union[float, None] = None,
        i_limit: Union[float, None] = None,
        dtype=jnp.float32,
    ) -> Tuple[NN_PIDState, Array]:
        """Compute NN-PID control action and update state.

        Returns ``(next_state, control_output)`` with control wrapped to shape (1,).
        """
        e = jnp.asarray(e, dtype=dtype)
        e = jnp.squeeze(e)

        dt_array = jnp.array(dt, dtype=dtype)

        # Accumulate integral error
        e_int = state.e_init + e * dt_array

        if i_limit is not None:
            lim = jnp.array(float(i_limit), dtype=dtype)
            e_int = jnp.clip(e_int, -lim, lim)

        de_dt = (e - state.e_prev) / dt_array

        nn_input = jnp.array([e, e_int, de_dt], dtype=dtype)
        u = NNPIDController._mlp_forward(nn_params, nn_input, hidden_activation=hidden_activation)

        if u_min is not None or u_max is not None:
            u = jnp.clip(u, a_min=u_min, a_max=u_max)

        next_state = NN_PIDState(e_init=e_int, e_prev=e)
        return next_state, jnp.array([u], dtype=dtype)

    def reset(self, state0: Optional[NN_PIDState] = None) -> NN_PIDState:
        """Reset controller to initial state."""
        if state0 is None:
            return self.create_initial_state(dtype=self.dtype)
        return NN_PIDState(
            e_init=jnp.array(state0.e_init, dtype=self.dtype),
            e_prev=jnp.array(state0.e_prev, dtype=self.dtype),
        )

    def step(
        self,
        state: NN_PIDState,
        y_ref: jnp.ndarray,
        y: jnp.ndarray,
        d: Union[jnp.ndarray, float] = 0.0,
    ) -> Tuple[NN_PIDState, jnp.ndarray]:
        """Compute one control step."""
        del d  # disturbance not directly used here
        e = jnp.asarray(y_ref - y, dtype=self.dtype)
        next_state, u = self.compute_control(
            self.params,
            state,
            e,
            self.dt,
            hidden_activation=self.hidden_activation,
            u_min=self.u_min,
            u_max=self.u_max,
            i_limit=self.i_limit,
            dtype=self.dtype,
        )
        return next_state, u

    def forward(self, x: Array) -> Array:
        """Run forward pass through neural network."""
        return self._mlp_forward(self.params, x, self.hidden_activation)

    def update_params(self, new_params) -> None:
        """Update neural network parameters (for training)."""
        self.params = new_params