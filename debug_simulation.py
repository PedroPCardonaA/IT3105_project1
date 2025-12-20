"""Debug script to see what's happening in the simulation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import yaml

from plants.bathtub import BathtubPlant, BathtubParams
from Controller.pid_controller import PIDController, PIDState

# Load config
config_path = Path(__file__).parent / "config" / "management.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Create plant
plant_params = BathtubParams(**config["plant"]["params"]["bathtub"])
plant = BathtubPlant(plant_params)

# Create controller
pid_config = config["controller"]["classic_pid"]
controller = PIDController(
    kp=pid_config["kp"],
    ki=pid_config["ki"],
    kd=pid_config["kd"],
    i_limit=pid_config["i_limit"]
)

# Simulation parameters
dt = config["simulation"]["dt"]
num_steps = 50  # Short simulation
reference = jnp.array(0.5)

# Initial states
state = plant.reset()
controller_state = PIDState(
    error_integral=jnp.array(0.0),
    prev_error=reference - state
)

print(f"Initial state: H = {state:.4f}")
print(f"Initial error: {controller_state.prev_error:.4f}")
print(f"Target: {reference:.4f}")
print(f"PID gains: Kp={controller.theta[0]:.4f}, Ki={controller.theta[1]:.4f}, Kd={controller.theta[2]:.4f}")
print("\nStep | H      | Error  | U      | dH/dt")
print("-" * 50)

for i in range(num_steps):
    # Compute error
    error = reference - state
    
    # Compute control
    u, controller_state = PIDController.compute_control(
        controller.theta, controller_state, error, dt
    )
    
    # Print every 10 steps
    if i % 10 == 0:
        # Compute derivative to see rate of change
        deriv = plant._deriv(state, u, 0.0)
        print(f"{i:4d} | {state:.4f} | {error:.4f} | {u:.4f} | {deriv:.4f}")
    
    # Step plant
    state = plant.step(state, u, 0.0)

print(f"\nFinal state: H = {state:.4f}")
print(f"Final error: {reference - state:.4f}")
print(f"MSE over trajectory: Would need to track all errors")
