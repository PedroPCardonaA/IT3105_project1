"""Main entry point for controller training and simulation."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml    
import jax
import jax.numpy as jnp

from src.plants.bathtub import BathtubPlant, BathtubParams
from src.plants.cournot import CournotPlant, CournotParams
from src.plants.dc_motor import DCMotorPlant, DCMotorParams
from src.Controller.pid_controller import PIDController
from src.Controller.nn_pid_controller import NNPIDController
from src.visualization import TrainingVisualizer
from src.training import train_controller
from src.results_exporter import ResultsExporter


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
        plant_params = BathtubParams(
            A=params['A'],
            C=params['C'],
            g=params['g'],
            Umin=params['Umin'],
            Umax=params['Umax'],
            Hmin=params['Hmin'],
        )
        plant = BathtubPlant(params=plant_params, dt=dt)
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
        plant = DCMotorPlant(params=plant_params, dt=dt, output='omega')
        initial_state = plant.reset(tuple(params['state0']))
        
    else:
        raise ValueError(f"Unknown plant type: {plant_type}")
    
    return plant, initial_state


def create_controller(config: Dict[str, Any], key: Optional[jax.Array] = None):
    """Create controller instance from configuration."""
    controller_type = config['controller']['type']
    dt = config['simulation']['dt']
    plant_type = config['plant']['type']
    plant_params = config['plant']['params'][plant_type]
    
    # Get plant-specific control limits
    u_min = plant_params.get('Umin')
    u_max = plant_params.get('Umax')
    
    if controller_type == 'classic':
        params = config['controller']['classic_pid']
        controller = PIDController(
            kp=params['kp'],
            ki=params['ki'],
            kd=params['kd'],
            dt=dt,
            i_limit=params.get('i_limit'),
            u_min=params.get('u_min', u_min),  # Use plant limits if not specified
            u_max=params.get('u_max', u_max),
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
            u_min=params.get('u_min', u_min),  # Use plant limits
            u_max=params.get('u_max', u_max),
            i_limit=params.get('i_limit', 5.0),
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
    
    # Reference setpoint from config
    plant_type = config['plant']['type']
    reference = config['plant']['params'][plant_type].get('reference', 1.0)
    
    # Visualization and results export
    viz = TrainingVisualizer()
    exporter = ResultsExporter()
    exporter.set_configuration(config)
    
    if verbose:
        print(f"Starting training with {plant_type} plant and {config['controller']['type']} controller")
        print(f"Epochs: {epochs}, Timesteps per epoch: {timesteps}")
        print(f"Reference setpoint: {reference}")
        print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        # Split key for randomness
        key, subkey = jax.random.split(key)
        
        # Generate random disturbances for this epoch
        d_min, d_max = disturbance_range
        disturbances = jax.random.uniform(
            subkey, shape=(timesteps,), minval=d_min, maxval=d_max
        )
        
        # Train controller using gradient descent
        controller, mse = train_controller(
            controller=controller,
            plant=plant,
            plant_state=plant_state,
            ctrl_state=ctrl_state,
            reference=reference,
            timesteps=timesteps,
            disturbances=disturbances,
            learning_rate=learning_rate,
            key=subkey,
        )
        
        # Run one epoch to get outputs for visualization
        errors, outputs, controls, plant_state, ctrl_state = run_epoch(
            plant, controller, plant_state, ctrl_state,
            reference, timesteps, disturbance_range, subkey
        )
        
        # Record metrics
        if config['controller']['type'] == 'classic' and isinstance(controller, PIDController):
            kp, ki, kd = float(controller.theta[0]), float(controller.theta[1]), float(controller.theta[2])
            viz.record_epoch(epoch, errors, kp=kp, ki=ki, kd=kd)
            avg_control = float(jnp.mean(controls))
            max_control = float(jnp.max(controls))
            exporter.add_epoch_result(
                epoch=epoch,
                mse=float(mse),
                final_output=float(outputs[-1]),
                avg_control=avg_control,
                max_control=max_control,
                kp=kp,
                ki=ki,
                kd=kd,
            )
        else:  # AI controller
            viz.record_epoch(epoch, errors)
            exporter.add_epoch_result(
                epoch=epoch,
                mse=float(mse),
                final_output=float(outputs[-1]),
            )
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            if isinstance(controller, PIDController):
                kp, ki, kd = controller.theta[0], controller.theta[1], controller.theta[2]
                avg_control = jnp.mean(controls)
                max_control = jnp.max(controls)
                print(f"Epoch {epoch:3d} | MSE: {mse:.6f} | Final H: {outputs[-1]:.4f} | Avg U: {avg_control:.4f} | Max U: {max_control:.4f} | Kp: {kp:.4f}, Ki: {ki:.4f}, Kd: {kd:.4f}")
            else:
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
    
    return viz, exporter, plant, controller, plant_state, ctrl_state


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
    viz, exporter, plant, controller, final_plant_state, final_ctrl_state = train(config, verbose=True)
    
    # Get summary for CSV export
    summary = viz.get_summary()
    
    # Create output directory structure
    plant_type = config['plant']['type']
    controller_type = config['controller']['type']
    output_dir = Path(args.output_dir) / plant_type / controller_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export CSV results (always)
    print(f"\nExporting results to: {output_dir}")
    exporter.export_all(output_dir, summary=summary)
    print(f"  Saved: {output_dir / 'configuration.csv'}")
    print(f"  Saved: {output_dir / 'training_results.csv'}")
    print(f"  Saved: {output_dir / 'training_summary.csv'}")
    
    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating plots...")
        
        # Learning progression plot (always)
        viz.plot_learning_progression(
            save_path=str(output_dir / "learning_progression.png")
        )
        print(f"  Saved: {output_dir / 'learning_progression.png'}")
        
        # PID parameters plot (only for classic controller)
        if controller_type == 'classic':
            viz.plot_pid_parameters(
                save_path=str(output_dir / "pid_parameters.png")
            )
            print(f"  Saved: {output_dir / 'pid_parameters.png'}")
            
            # Combined plot
            viz.plot_combined(
                save_path=str(output_dir / "training_summary.png")
            )
            print(f"  Saved: {output_dir / 'training_summary.png'}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
