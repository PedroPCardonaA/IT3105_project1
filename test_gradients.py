"""Test gradient computation through PID controller."""

import jax
import jax.numpy as jnp
import jax.lax as lax

from src.Controller.pid_controller import PIDController, PIDState
from plants.bathtub import BathtubPlant, BathtubParams


def test_simple_gradient():
    """Test if gradients flow through a simple PID computation."""
    # Simple loss: output of PID control squared
    def simple_loss(theta):
        kp, ki, kd = theta[0], theta[1], theta[2]
        error = jnp.array([1.0])
        state = PIDState(e_init=jnp.array(0.0), e_prev=jnp.array(0.0))
        
        next_state, u = PIDController.compute_control(
            theta=theta,
            state=state,
            e=error,
            dt=0.01,
        )
        return jnp.sum(u ** 2)
    
    theta = jnp.array([0.5, 0.2, 0.05])
    grad_fn = jax.grad(simple_loss)
    grads = grad_fn(theta)
    print(f"Simple gradient test:")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print(f"  Gradient norm: {jnp.linalg.norm(grads)}")
    print()


def test_scan_gradient():
    """Test if gradients flow through lax.scan."""
    def scan_loss(theta):
        def step_fn(carry, x):
            return carry + theta[0] * x, carry + theta[0] * x
        
        _, outputs = lax.scan(step_fn, 0.0, jnp.array([1.0, 2.0, 3.0]))
        return jnp.sum(outputs ** 2)
    
    theta = jnp.array([0.5, 0.2, 0.05])
    grad_fn = jax.grad(scan_loss)
    grads = grad_fn(theta)
    print(f"Scan gradient test:")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print(f"  Gradient norm: {jnp.linalg.norm(grads)}")
    print()


def test_pid_through_plant():
    """Test gradients through PID + plant."""
    # Create plant
    params = BathtubParams(A=1.0, C=0.5, g=9.81)
    plant = BathtubPlant(params, dt=0.01)
    
    # Test: Plant step with constant control
    def plant_loss(u_val):
        plant_state = jnp.array([0.2])
        u = jnp.array([u_val])
        next_plant_state, _ = plant.step(plant_state, u, d=0.0)
        next_y = plant.output(next_plant_state)
        return jnp.sum((jnp.array([0.5]) - next_y) ** 2)
    
    grad_fn = jax.grad(plant_loss)
    grad = grad_fn(0.01)
    print(f"Plant-only gradient (wrt control):")
    print(f"  Control u: 0.01")
    print(f"  Gradient: {grad}")
    print()
    
    # Test: Full PID + Plant (simplified - just use control u directly)
    def loss_fn_super_simple(theta):
        plant_state = jnp.array([0.2])
        
        # Simple proportional control only
        kp = theta[0]
        error = 0.3  # Fixed error
        u = kp * error
        u_array = jnp.array([u])
        
        next_plant_state, _ = plant.step(plant_state, u_array, d=0.0)
        # Return next state directly
        return jnp.sum(next_plant_state ** 2)
    
    theta = jnp.array([0.5, 0.2, 0.05])
    grad_fn = jax.grad(loss_fn_super_simple)
    grads = grad_fn(theta)
    print(f"Super simplified test (P control only, next_state loss):")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test: With output function
    def loss_fn_with_output(theta):
        plant_state = jnp.array([0.2])
        
        kp = theta[0]
        error = 0.3
        u = kp * error
        u_array = jnp.array([u])
        
        next_plant_state, _ = plant.step(plant_state, u_array, d=0.0)
        next_y = plant.output(next_plant_state)  # Use output function
        return jnp.sum(next_y ** 2)
    
    grad_fn = jax.grad(loss_fn_with_output)
    grads = grad_fn(theta)
    print(f"With plant.output():")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test: Without using plant instance methods
    def loss_direct(theta):
        from plants.bathtub import _deriv
        from src.utils import rk4_step
        
        plant_state = jnp.array([0.2])
        kp = theta[0]
        error = 0.3
        u = kp * error
        u_array = jnp.array([u])
        
        # Directly call rk4_step
        next_state = rk4_step(_deriv, params, plant_state, u_array, 0.0, 0.01)
        return jnp.sum(next_state ** 2)
    
    grad_fn = jax.grad(loss_direct)
    grads = grad_fn(theta)
    print(f"Direct rk4_step (no plant instance):")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test: Full PID + Plant (using compute_control)
    def loss_fn(theta):
        plant_state = jnp.array([0.2])
        ctrl_state = PIDState(e_init=jnp.array(0.0), e_prev=jnp.array(0.0))
        
        y = plant.output(plant_state)
        error = jnp.array([0.5]) - y
        
        next_ctrl_state, u = PIDController.compute_control(
            theta=theta,
            state=ctrl_state,
            e=error,
            dt=0.01,
        )
        
        next_plant_state, _ = plant.step(plant_state, u, d=0.0)
        next_y = plant.output(next_plant_state)
        next_error = jnp.array([0.5]) - next_y
        return jnp.sum(next_error ** 2)
    
    theta = jnp.array([0.5, 0.2, 0.05])
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(theta)
    print(f"PID + Plant gradient test (using compute_control):")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print(f"  Gradient norm: {jnp.linalg.norm(grads)}")
    print()


if __name__ == "__main__":
    test_simple_gradient()
    test_scan_gradient()
    test_pid_through_plant()
