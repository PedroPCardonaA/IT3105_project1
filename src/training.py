"""Training utilities for PID and NN-PID controllers using JAX gradient descent."""

from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp

from .Controller.base_controller import ControllerBase
from .Controller.pid_controller import PIDController, PIDState
from .Controller.nn_pid_controller import NNPIDController, NN_PIDState


Array = jnp.ndarray


def compute_mse_loss(errors: Array) -> Array:
    """Compute Mean Squared Error loss from tracking errors."""
    return jnp.mean(errors ** 2)


def run_epoch_for_loss(
    plant,
    controller,
    plant_state: Array,
    ctrl_state: Union[PIDState, NN_PIDState],
    reference: float,
    timesteps: int,
    disturbances: Array,
) -> Tuple[Array, Array, Array]:
    """
    Run one epoch and return errors, outputs, and controls.
    
    This is a pure functional version for use inside JAX gradient computations.
    """
    errors = []
    outputs = []
    controls = []
    
    current_plant_state = plant_state
    current_ctrl_state = ctrl_state
    
    for t in range(timesteps):
        # Get current output
        y = plant.output(current_plant_state)
        outputs.append(y)
        
        # Compute error
        y_ref = jnp.array([reference])
        error = y_ref - y
        errors.append(error)
        
        # Controller step
        current_ctrl_state, u = controller.step(
            current_ctrl_state, y_ref, y, d=disturbances[t]
        )
        controls.append(u)
        
        # Plant step
        current_plant_state, _ = plant.step(
            current_plant_state, u, d=disturbances[t]
        )
    
    errors_array = jnp.array(errors).squeeze()
    outputs_array = jnp.array(outputs).squeeze()
    controls_array = jnp.array(controls).squeeze()
    
    return errors_array, outputs_array, controls_array


def create_pid_loss_function(
    plant_step_fn: Callable,
    plant_output_fn: Callable,
    plant_state: Array,
    ctrl_state: PIDState,
    reference: float,
    timesteps: int,
    disturbances: Array,
    dt: float,
    i_limit: Union[float, None],
    u_min: Union[float, None],
    u_max: Union[float, None],
    dtype=jnp.float32,
) -> Callable[[Array], Array]:
    """
    Create a loss function for classic PID controller that takes theta as input.
    Uses lax.scan for efficient JAX compilation.
    
    Args:
        plant_step_fn: Pure function (state, u, d) -> (next_state, output)
        plant_output_fn: Pure function (state) -> output
    """
    # Make step and output functions static by wrapping in jax.pure_callback? No, let's try staticmethod
    # Actually, the issue is these ARE closures. Let me try a different approach.
    # The plant_step_fn and plant_output_fn are defined in the update function and capture `plant`
    # We need to make them truly pure with no captures
    
    def loss_fn(theta: Array) -> Array:
        # Define the step function that uses theta as a scanned variable
        def pid_step(carry, inputs):
            plant_st, ctrl_st = carry
            disturbance, theta_i = inputs  # theta passed as input to scan
            
            # Get current output (functional call)
            y = plant_output_fn(plant_st)
            
            # Compute error
            y_ref = jnp.array([reference], dtype=dtype)
            error = y_ref - y
            
            # Functional PID control computation
            next_ctrl_st, u = PIDController.compute_control(
                theta=theta_i,
                state=ctrl_st,
                e=error,
                dt=dt,
                u_min=u_min,
                u_max=u_max,
                i_limit=i_limit,
                dtype=dtype,
            )
            
            # Plant step (functional call)
            next_plant_st, _ = plant_step_fn(plant_st, u, disturbance)
            
            return (next_plant_st, next_ctrl_st), error
        
        # Broadcast theta to all timesteps
        theta_broadcast = jnp.tile(theta[None, :], (timesteps, 1))
        inputs = (disturbances, theta_broadcast)
        
        # Run scan over all timesteps
        _, errors = lax.scan(pid_step, (plant_state, ctrl_state), inputs)
        
        # Compute and return MSE loss
        return compute_mse_loss(errors.squeeze())
    
    # Don't wrap in jax.jit here - let the caller decide
    return loss_fn


def update_pid_controller(
    controller: PIDController,
    plant,
    plant_state: Array,
    ctrl_state: PIDState,
    reference: float,
    timesteps: int,
    disturbances: Array,
    learning_rate: float,
) -> Tuple[PIDController, float]:
    """
    Update PID controller parameters using gradient descent.
    
    This function uses the plant's step method directly, making it plant-agnostic.
    """
    # Create a pure loss function
    def loss_fn(theta: Array) -> Array:
        def pid_step(carry, inputs):
            plant_st, ctrl_st = carry
            disturbance, theta_i = inputs
            
            # Get output using plant's output method
            y = plant.output(plant_st)
            
            # Compute error
            y_ref = jnp.array([reference], dtype=controller.dtype)
            error = y_ref - y
            
            # PID control
            next_ctrl_st, u = PIDController.compute_control(
                theta=theta_i,
                state=ctrl_st,
                e=error,
                dt=controller.dt,
                u_min=controller.u_min,
                u_max=controller.u_max,
                i_limit=controller.i_limit,
                dtype=controller.dtype,
            )
            
            # Use plant's step method (plant-agnostic)
            next_plant_st, _ = plant.step(plant_st, u, disturbance)
            
            return (next_plant_st, next_ctrl_st), error
        
        # Broadcast theta
        theta_broadcast = jnp.tile(theta[None, :], (timesteps, 1))
        inputs = (disturbances, theta_broadcast)
        
        # Run scan
        _, errors = lax.scan(pid_step, (plant_state, ctrl_state), inputs)
        
        return compute_mse_loss(errors.squeeze())
    
    # Compute loss and gradient using JAX autodiff
    loss_value = loss_fn(controller.theta)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(controller.theta)
    
    # Check for NaN/Inf in gradients
    if jnp.any(jnp.isnan(grads)) or jnp.any(jnp.isinf(grads)):
        print(f"Warning: NaN or Inf gradients detected: {grads}")
        print(f"  Loss value: {loss_value}")
        print(f"  Theta: {controller.theta}")
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update parameters using gradient descent
    new_theta = controller.theta - learning_rate * grads
    
    # Create new controller with updated parameters
    updated_controller = PIDController(
        kp=float(new_theta[0]),
        ki=float(new_theta[1]),
        kd=float(new_theta[2]),
        dt=controller.dt,
        i_limit=controller.i_limit,
        u_min=controller.u_min,
        u_max=controller.u_max,
    )
    
    return updated_controller, float(loss_value)


def create_nn_pid_loss_function(
    plant,
    plant_state: Array,
    ctrl_state: NN_PIDState,
    reference: float,
    timesteps: int,
    disturbances: Array,
    dt: float,
    hidden_activation: Any,
    i_limit: Union[float, None],
    u_min: Union[float, None],
    u_max: Union[float, None],
    key: Array,
    dtype=jnp.float32,
) -> Callable[[list], Array]:
    """
    Create an optimized loss function for NN-PID controller using lax.scan.
    
    Args:
        plant: Plant instance
        plant_state: Initial plant state
        ctrl_state: Initial controller state
        reference: Reference setpoint
        timesteps: Number of timesteps per epoch
        disturbances: Array of disturbances for this epoch
        dt: Time step
        hidden_activation: Activation function name
        i_limit: Integral limit
        u_min: Minimum control output
        u_max: Maximum control output
        key: Random key for initialization
        dtype: JAX dtype
    
    Returns:
        A function that takes neural network params and returns MSE loss
    """
    def loss_fn(params: list) -> Array:
        # Define step function for lax.scan
        def nn_pid_step(carry, inputs):
            plant_st, ctrl_st = carry
            disturbance, params_i = inputs
            
            # Get output using plant's output method
            y = plant.output(plant_st)
            
            # Compute error
            y_ref = jnp.array([reference], dtype=dtype)
            error = y_ref - y
            
            # NN-PID control using static method
            next_ctrl_st, u = NNPIDController.compute_control(
                nn_params=params_i,
                state=ctrl_st,
                e=error,
                dt=dt,
                hidden_activation=hidden_activation,
                u_min=u_min,
                u_max=u_max,
                i_limit=i_limit,
                dtype=dtype,
            )
            
            # Use plant's step method (plant-agnostic)
            next_plant_st, _ = plant.step(plant_st, u, disturbance)
            
            return (next_plant_st, next_ctrl_st), error
        
        # Broadcast params to all timesteps (needed for scan)
        # Convert params list to a structure that can be broadcast
        params_broadcast = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (timesteps,) + (1,) * x.ndim),
            params
        )
        
        inputs = (disturbances, params_broadcast)
        
        # Run scan over all timesteps
        _, errors = lax.scan(nn_pid_step, (plant_state, ctrl_state), inputs)
        
        # Compute and return MSE loss
        return compute_mse_loss(errors.squeeze())
    
    return loss_fn


def update_nn_pid_controller(
    controller: NNPIDController,
    plant,
    plant_state: Array,
    ctrl_state: NN_PIDState,
    reference: float,
    timesteps: int,
    disturbances: Array,
    learning_rate: float,
    key: Array,
) -> Tuple[NNPIDController, float]:
    """
    Update NN-PID controller parameters using gradient descent.
    
    Args:
        controller: Current NN-PID controller
        plant: Plant instance
        plant_state: Initial plant state
        ctrl_state: Initial controller state
        reference: Reference setpoint
        timesteps: Number of timesteps per epoch
        disturbances: Array of disturbances for this epoch
        learning_rate: Learning rate for gradient descent
        key: Random key
    
    Returns:
        Updated controller and the loss value
    """
    # Create loss function
    loss_fn = create_nn_pid_loss_function(
        plant=plant,
        plant_state=plant_state,
        ctrl_state=ctrl_state,
        reference=reference,
        timesteps=timesteps,
        disturbances=disturbances,
        dt=controller.dt,
        hidden_activation=controller.hidden_activation,
        i_limit=controller.i_limit,
        u_min=controller.u_min,
        u_max=controller.u_max,
        key=key,
        dtype=controller.dtype,
    )
    
    # Compute loss and gradient
    loss_value = loss_fn(controller.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(controller.params)
    
    # Update parameters using gradient descent
    new_params = []
    for layer_grad, layer_param in zip(grads, controller.params):
        new_W = layer_param['W'] - learning_rate * layer_grad['W']
        new_b = layer_param['b'] - learning_rate * layer_grad['b']
        new_params.append({'W': new_W, 'b': new_b})
    
    # Update controller params in place
    controller.update_params(new_params)
    
    return controller, float(loss_value)


def train_controller(
    controller: Union[PIDController, NNPIDController],
    plant,
    plant_state: Array,
    ctrl_state: Union[PIDState, NN_PIDState],
    reference: float,
    timesteps: int,
    disturbances: Array,
    learning_rate: float,
    key: Array,
) -> Tuple[Union[PIDController, NNPIDController], float]:
    """
    Unified function to train either PID or NN-PID controller.
    
    Args:
        controller: Controller instance (PID or NN-PID)
        plant: Plant instance
        plant_state: Initial plant state
        ctrl_state: Initial controller state
        reference: Reference setpoint
        timesteps: Number of timesteps per epoch
        disturbances: Array of disturbances for this epoch
        learning_rate: Learning rate for gradient descent
        key: Random key
    
    Returns:
        Updated controller and the loss value
    """
    if isinstance(controller, PIDController):
        # Type narrowing for PID controller
        assert isinstance(ctrl_state, PIDState)
        return update_pid_controller(
            controller=controller,
            plant=plant,
            plant_state=plant_state,
            ctrl_state=ctrl_state,
            reference=reference,
            timesteps=timesteps,
            disturbances=disturbances,
            learning_rate=learning_rate,
        )
    elif isinstance(controller, NNPIDController):
        # Type narrowing for NN-PID controller
        assert isinstance(ctrl_state, NN_PIDState)
        return update_nn_pid_controller(
            controller=controller,
            plant=plant,
            plant_state=plant_state,
            ctrl_state=ctrl_state,
            reference=reference,
            timesteps=timesteps,
            disturbances=disturbances,
            learning_rate=learning_rate,
            key=key,
        )
    else:
        raise ValueError(f"Unknown controller type: {type(controller)}")
