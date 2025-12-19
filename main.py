"""Main entry point for controller training and simulation."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml    
import jax
import jax.numpy as jnp

from src.plants.bathub import BathubPlant, BathubParams
from src.plants.cournot import CournotPlant, CournotParams
from src.plants.dc_motor import DCMotorPlant, DCMotorParams
from src.Controller.pid_controller import PIDController
from src.Controller.nn_pid_controller import NNPIDController
from src.visualization import TrainingVisualizer


def load_config(config_path: str = "config/management.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_plant(config: Dict[str, Any]):
    """Create plant instance from configuration."""
    plant_type = config['plant']['type']
    params = config['plant']['params'][plant_type]
    dt = config['simulation']['dt']
    
    if plant_type == 'bathtub':
        plant_params = BathubParams(
            A=params['A'],
            C=params['C'],
            g=params['g'],
            Umin=params['Umin'],
            Umax=params['Umax'],
            Hmin=params['Hmin'],
        )
        plant = BathubPlant(params=plant_params, dt=dt)
        initial_state = plant.reset(params['H0'])
        
    elif plant_type == 'cournot':
        plant_params = CournotParams(
            pmax=params['pmax'],
            cm=params['cm'],
            qmin=params['qmin'],
            qmax=params['qmax'],
            Umin=params['Umin'],
            Umax=params['Umax'],
        )
        plant = CournotPlant(params=plant_params, dt=dt)
        initial_state = plant.reset(tuple(params['q0']))
        
    elif plant_type == 'dc_motor':
        plant_params = DCMotorParams(
            R=params['R'],
            L=params['L'],
            Ke=params['Ke'],
            Kt=params['Kt'],
            J=params['J'],
            b=params['b'],
            tau_c=params['tau_c'],
            omega_s=params['omega_s'],
            tau1=params['tau1'],
            Vmax=params['Vmax'],
        )
        plant = DCMotorPlant(params=plant_params, dt=dt)
        initial_state = plant.reset(tuple(params['state0']))
        
    else:
        raise ValueError(f"Unknown plant type: {plant_type}")
    
    return plant, initial_state


def create_controller(config: Dict[str, Any], key: Optional[jax.Array] = None):
    """Create controller instance from configuration."""
    controller_type = config['controller']['type']
    dt = config['simulation']['dt']
    
    if controller_type == 'classic':
        params = config['controller']['classic_pid']
        controller = PIDController(
            kp=params['kp'],
            ki=params['ki'],
            kd=params['kd'],
            dt=dt,
            i_limit=params.get('i_limit'),
            u_min=params.get('u_min'),
            u_max=params.get('u_max'),
        )
        initial_ctrl_state = controller.reset()
        
    elif controller_type == 'ai':
        params = config['controller']['ai']
        
        # Build layer sizes: [input=3, hidden_layers..., output=1]
        layer_sizes = [3] + params['hidden_layer_sizes'] + [1]
        
        # Use first activation for all hidden layers (simplification)
        # You can extend this to use different activations per layer
        hidden_activation = params['activations'][0]
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        controller = NNPIDController(
            layer_sizes=tuple(layer_sizes),
            hidden_activation=hidden_activation,
            dt=dt,
            w_init_range=tuple(params['weight_init_range']),
            b_init_range=tuple(params['bias_init_range']),
            key=key,
        )
        initial_ctrl_state = controller.reset()
        
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    return controller, initial_ctrl_state


def run_epoch(
    plant,
    controller,
    plant_state: jnp.ndarray,
    ctrl_state,
    reference: float,
    timesteps: int,
    disturbance_range: tuple,
    key: jax.Array,
) -> tuple:
    """Run one training epoch."""
    # Generate random disturbances
    d_min, d_max = disturbance_range
    disturbances = jax.random.uniform(
        key, shape=(timesteps,), minval=d_min, maxval=d_max
    )
    
    # Storage
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
    
    return errors_array, outputs_array, controls_array, current_plant_state, current_ctrl_state


def train(config: Dict[str, Any], verbose: bool = True):
    """Main training loop."""
    # Set random seed
    key = jax.random.PRNGKey(0)
    
    # Create plant and controller
    plant, plant_state = create_plant(config)
    controller, ctrl_state = create_controller(config, key)
    
    # Training parameters
    epochs = config['training']['epochs']
    timesteps = config['training']['timesteps_per_epoch']
    learning_rate = config['training']['learning_rate']
    disturbance_range = tuple(config['simulation']['disturbance_range'])
    
    # Reference setpoint (you can make this configurable)
    plant_type = config['plant']['type']
    if plant_type == 'bathtub':
        reference = 0.5  # Target water height
    elif plant_type == 'cournot':
        reference = 50.0  # Target profit or quantity
    elif plant_type == 'dc_motor':
        reference = 1.0  # Target angular velocity
    else:
        reference = 1.0
    
    # Visualization
    viz = TrainingVisualizer()
    
    if verbose:
        print(f"Starting training with {plant_type} plant and {config['controller']['type']} controller")
        print(f"Epochs: {epochs}, Timesteps per epoch: {timesteps}")
        print(f"Reference setpoint: {reference}")
        print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        # Split key for randomness
        key, subkey = jax.random.split(key)
        
        # Run one epoch
        errors, outputs, controls, plant_state, ctrl_state = run_epoch(
            plant, controller, plant_state, ctrl_state,
            reference, timesteps, disturbance_range, subkey
        )
        
        # Compute MSE
        mse = float(jnp.mean(errors ** 2))
        
        # Record metrics
        if config['controller']['type'] == 'classic' and isinstance(controller, PIDController):
            kp, ki, kd = float(controller.theta[0]), float(controller.theta[1]), float(controller.theta[2])
            viz.record_epoch(epoch, errors, kp=kp, ki=ki, kd=kd)
            
            # Simple gradient-based parameter update (optional - basic tuning)
            # This is a simplified version; you can implement proper gradient descent
            # For now, we'll keep parameters fixed as specified in config
            
        else:  # AI controller
            viz.record_epoch(epoch, errors)
            # Here you would implement neural network training
            # using JAX autograd to compute gradients and update params
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | MSE: {mse:.6f} | Final output: {outputs[-1]:.4f}")
    
    if verbose:
        print("-" * 60)
        summary = viz.get_summary()
        print("Training Summary:")
        print(f"  Final MSE: {summary['final_mse']:.6f}")
        print(f"  Best MSE: {summary['best_mse']:.6f}")
        print(f"  Average MSE: {summary['average_mse']:.6f}")
        
        if config['controller']['type'] == 'classic':
            print(f"  Final PID gains: Kp={summary['final_kp']:.4f}, Ki={summary['final_ki']:.4f}, Kd={summary['final_kd']:.4f}")
    
    return viz, plant, controller, plant_state, ctrl_state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train and simulate control systems')
    parser.add_argument(
        '--config',
        type=str,
        default='config/management.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save output plots'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    viz, plant, controller, final_plant_state, final_ctrl_state = train(config, verbose=True)
    
    # Generate plots if requested
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\nGenerating plots...")
        
        # Learning progression plot (always)
        viz.plot_learning_progression(
            save_path=str(output_dir / "learning_progression.png")
        )
        print(f"  Saved: {output_dir / 'learning_progression.png'}")
        
        # PID parameters plot (only for classic controller)
        if config['controller']['type'] == 'classic':
            viz.plot_pid_parameters(
                save_path=str(output_dir / "pid_parameters.png")
            )
            print(f"  Saved: {output_dir / 'pid_parameters.png'}")
            
            # Combined plot
            viz.plot_combined(
                save_path=str(output_dir / "training_summary.png")
            )
            print(f"  Saved: {output_dir / 'training_summary.png'}")
        
        print("Done!")


if __name__ == "__main__":
    main()
