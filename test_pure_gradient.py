"""Test if pure functional approach gives non-zero gradients."""

import jax
import jax.numpy as jnp
import jax.lax as lax

from src.Controller.pid_controller import PIDController, PIDState
from plants.bathtub import BathtubPlant, BathtubParams, _deriv
from src.utils import rk4_step


def test_pure_functional_gradient():
    """Test gradients with pure functional plant dynamics."""
    # Create plant params directly
    params = BathtubParams(A=1.0, C=0.5, g=9.81)
    plant_dt = 0.01
    plant_dtype = jnp.float32
    
    # Test 1: Just the plant dynamics
    def plant_only_loss(u_val):
        plant_state = jnp.array([0.2], dtype=plant_dtype)
        u = jnp.array([u_val], dtype=plant_dtype)
        d = jnp.array(0.0, dtype=plant_dtype)
        next_state = rk4_step(_deriv, params, plant_state, u, d, plant_dt)
        return jnp.sum((jnp.array([0.5]) - next_state[0:1]) ** 2)
    
    grad_fn = jax.grad(plant_only_loss)
    grad = grad_fn(0.01)
    print(f"Plant-only gradient (wrt u):")
    print(f"  u: 0.01")
    print(f"  Gradient: {grad}")
    print()
    
    # Test 2: PID output only
    def pid_output_loss(theta):
        ctrl_state = PIDState(e_init=jnp.array(0.0), e_prev=jnp.array(0.0))
        error = jnp.array([0.3])
        _, u = PIDController.compute_control(
            theta=theta,
            state=ctrl_state,
            e=error,
            dt=0.01,
        )
        return jnp.sum(u ** 2)
    
    theta = jnp.array([0.5, 0.2, 0.05])
    grad_fn = jax.grad(pid_output_loss)
    grads = grad_fn(theta)
    print(f"PID-only gradient:")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test clip with different values
    def test_clip_middle(theta):
        u = theta[0] * 0.01  # Should be well within [0, 0.05]
        print(f"    u before clip: {u}")
        u_clipped = jnp.clip(u, 0.0, 0.05)
        print(f"    u after clip: {u_clipped}")
        return u_clipped ** 2
    
    print("Testing clip (u should be ~0.005, within [0, 0.05]):")
    grad_fn = jax.grad(test_clip_middle)
    grads = grad_fn(theta)
    print(f"  Gradients: {grads}")
    print()
    
    def test_clip_original(theta):
        u = theta[0] * 0.3  # 0.5 * 0.3 = 0.15, OUTSIDE [0, 0.05]!
        print(f"    u before clip: {u}")
        u_clipped = jnp.clip(u, 0.0, 0.05)
        print(f"    u after clip: {u_clipped}")
        return u_clipped ** 2
    
    print("Testing clip (u = 0.15, OUTSIDE [0, 0.05]):")
    grad_fn = jax.grad(test_clip_original)
    grads = grad_fn(theta)
    print(f"  Gradients: {grads}")
    print()
    
    def test_sqrt(theta):
        u = theta[0] * 0.3
        result = jnp.sqrt(u)
        return result ** 2
    
    grad_fn = jax.grad(test_sqrt)
    grads = grad_fn(theta)
    print(f"With sqrt:")
    print(f"  Gradients: {grads}")
    print()
    
    def test_single_plant_op(theta):
        u = theta[0] * 0.3
        H = 0.2
        H_pos = jnp.maximum(H, 0.0)
        V = jnp.sqrt(2 * 9.81 * H_pos)
        Q = 0.5 * V
        dH = (u - Q) / 1.0
        return dH ** 2
    
    grad_fn = jax.grad(test_single_plant_op)
    grads = grad_fn(theta)
    print(f"Single plant-like operation:")
    print(f"  Gradients: {grads}")
    print()
    def absolute_simplest(theta):
        return theta[0] ** 2
    
    grad_fn = jax.grad(absolute_simplest)
    grads = grad_fn(theta)
    print(f"Absolute simplest (theta[0]**2):")
    print(f"  Gradients: {grads}")
    print()
    
    # Test: theta -> array operations -> output
    def with_array_ops(theta):
        u = jnp.array([theta[0] * 0.3])
        return jnp.sum(u ** 2)
    
    grad_fn = jax.grad(with_array_ops)
    grads = grad_fn(theta)
    print(f"With array ops:")
    print(f"  Gradients: {grads}")
    print()
    def ultra_simple_unpacked(theta):
        kp = theta[0]
        u_scalar = kp * 0.3
        
        # Unpack params into plain values
        A, C, g = params.A, params.C, params.g
        Hmin, Umin, Umax = params.Hmin, params.Umin, params.Umax
        
        # Define deriv inline without dataclass
        def deriv_inline(state, u, d):
            H = state[0]
            U = jnp.clip(u[0], Umin, Umax)
            H_pos = jnp.maximum(H, 0.0)
            V = jnp.sqrt(2 * g * H_pos)
            Q = C * V
            dH = (U - Q + d) / A
            return jnp.array([dH])
        
        # Simple Euler step (not rk4 for simplicity)
        plant_state = jnp.array([0.2])
        u = jnp.array([u_scalar])
        d = jnp.array(0.0)
        
        dstate = deriv_inline(plant_state, u, d)
        next_state = plant_state + plant_dt * dstate
        
        return next_state[0]
    
    grad_fn = jax.grad(ultra_simple_unpacked)
    grads = grad_fn(theta)
    print(f"Ultra simple UNPACKED (no dataclass):")
    print(f"  Gradients: {grads}")
    print()
    def ultra_simple(theta):
        kp = theta[0]
        u_scalar = kp * 0.3  # Just a scalar
        
        # Convert to array for plant
        plant_state = jnp.array([0.2])
        u = jnp.array([u_scalar])
        d = jnp.array(0.0)
        next_state = rk4_step(_deriv, params, plant_state, u, d, plant_dt)
        
        return next_state[0]  # Return scalar
    
    grad_fn = jax.grad(ultra_simple)
    grads = grad_fn(theta)
    print(f"Ultra simple (theta -> u_scalar -> plant):")
    print(f"  Gradients: {grads}")
    print()
    def manual_loss_direct_state(theta):
        plant_state = jnp.array([0.2], dtype=plant_dtype)
        
        # Manual PID (no state)
        kp = theta[0]
        error = 0.3
        u_val = kp * error
        
        # Plant step
        u = jnp.array([u_val], dtype=plant_dtype)
        d = jnp.array(0.0, dtype=plant_dtype)
        next_state = rk4_step(_deriv, params, plant_state, u, d, plant_dt)
        
        # Loss is just next state value (not error)
        return jnp.sum(next_state ** 2)
    
    grad_fn = jax.grad(manual_loss_direct_state)
    grads = grad_fn(theta)
    print(f"Manual P-control + Plant (next_state loss):")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test 3b: With squared error loss
    def manual_loss(theta):
        plant_state = jnp.array([0.2], dtype=plant_dtype)
        
        # Manual PID (no state)
        kp = theta[0]
        error = 0.3
        u_val = kp * error
        
        # Plant step
        u = jnp.array([u_val], dtype=plant_dtype)
        d = jnp.array(0.0, dtype=plant_dtype)
        next_state = rk4_step(_deriv, params, plant_state, u, d, plant_dt)
        
        return jnp.sum((jnp.array([0.5]) - next_state[0:1]) ** 2)
    
    grad_fn = jax.grad(manual_loss)
    grads = grad_fn(theta)
    print(f"Manual P-control + Plant (error loss):")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print()
    
    # Test 4: Full chain
    def loss_fn(theta):
        plant_state = jnp.array([0.2], dtype=plant_dtype)
        ctrl_state = PIDState(e_init=jnp.array(0.0), e_prev=jnp.array(0.0))
        
        # Output
        y = plant_state[0:1]
        error = jnp.array([0.5]) - y
        
        # PID control
        _, u = PIDController.compute_control(
            theta=theta,
            state=ctrl_state,
            e=error,
            dt=0.01,
        )
        
        # Plant step
        u_typed = jnp.array(u, dtype=plant_dtype)
        d_typed = jnp.array(0.0, dtype=plant_dtype)
        next_state = rk4_step(_deriv, params, plant_state, u_typed, d_typed, plant_dt)
        
        # Next error
        next_y = next_state[0:1]
        next_error = jnp.array([0.5]) - next_y
        
        return jnp.sum(next_error ** 2)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(theta)
    
    print(f"Full PID + Plant:")
    print(f"  Theta: {theta}")
    print(f"  Gradients: {grads}")
    print(f"  Gradient norm: {jnp.linalg.norm(grads)}")
    
    if jnp.linalg.norm(grads) > 0:
        print("  ✓ SUCCESS: Non-zero gradients!")
    else:
        print("  ✗ FAIL: Zero gradients")


if __name__ == "__main__":
    test_pure_functional_gradient()
