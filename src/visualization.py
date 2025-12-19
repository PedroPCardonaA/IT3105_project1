"""Visualization utilities for controller training and performance assessment."""

from __future__ import annotations
from typing import List, Optional, Tuple

import jax.numpy as jnp

from .utils import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.figure import Figure



class TrainingVisualizer:
    """Track and visualize controller training progress."""

    def __init__(self):
        """Initialize tracking lists for metrics."""
        self.epochs: List[int] = []
        self.mse_values: List[float] = []
        self.kp_values: List[float] = []
        self.ki_values: List[float] = []
        self.kd_values: List[float] = []

    def record_epoch(
        self,
        epoch: int,
        errors: jnp.ndarray,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> None:
        """Record metrics for a training epoch.

        Args:
            epoch: Current epoch number.
            errors: Array of error values for this epoch.
            kp: Proportional gain (PID only).
            ki: Integral gain (PID only).
            kd: Derivative gain (PID only).
        """
        self.epochs.append(epoch)
        
        # Compute and store MSE for this epoch
        mse = float(mean_squared_error(errors, jnp.zeros_like(errors)))
        self.mse_values.append(mse)

        # Store PID parameters if provided
        if kp is not None:
            self.kp_values.append(kp)
        if ki is not None:
            self.ki_values.append(ki)
        if kd is not None:
            self.kd_values.append(kd)

    def record_mse(self, epoch: int, mse: float) -> None:
        """Record MSE directly for an epoch.

        Args:
            epoch: Current epoch number.
            mse: Mean squared error value.
        """
        self.epochs.append(epoch)
        self.mse_values.append(mse)

    def record_pid_params(self, kp: float, ki: float, kd: float) -> None:
        """Record PID parameters for current epoch.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
        """
        self.kp_values.append(kp)
        self.ki_values.append(ki)
        self.kd_values.append(kd)

    def plot_learning_progression(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Learning Progression",
        save_path: Optional[str] = None,
    ) -> "Figure":
        """Plot MSE vs epoch number.

        Args:
            figsize: Figure size (width, height).
            title: Plot title.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.epochs, self.mse_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_pid_parameters(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "PID Parameter Evolution",
        save_path: Optional[str] = None,
    ) -> "Figure":
        """Plot PID parameter changes across epochs.

        Args:
            figsize: Figure size (width, height).
            title: Plot title.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.

        Raises:
            ValueError: If no PID parameters have been recorded.
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        if not self.kp_values or not self.ki_values or not self.kd_values:
            raise ValueError("No PID parameters recorded. Call record_pid_params() first.")

        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = self.epochs[:len(self.kp_values)]
        
        ax.plot(epochs, self.kp_values, 'r-', linewidth=2, marker='o', markersize=4, label='Kp')
        ax.plot(epochs, self.ki_values, 'g-', linewidth=2, marker='s', markersize=4, label='Ki')
        ax.plot(epochs, self.kd_values, 'b-', linewidth=2, marker='^', markersize=4, label='Kd')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_combined(
        self,
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None,
    ) -> "Figure":
        """Plot both learning progression and PID parameters side by side.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.

        Raises:
            ValueError: If no PID parameters have been recorded.
        """
        if plt is None:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        if not self.kp_values or not self.ki_values or not self.kd_values:
            raise ValueError("No PID parameters recorded. This plot requires PID data.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Learning progression
        ax1.plot(self.epochs, self.mse_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
        ax1.set_title('Learning Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        
        # Right plot: PID parameters
        epochs = self.epochs[:len(self.kp_values)]
        ax2.plot(epochs, self.kp_values, 'r-', linewidth=2, marker='o', markersize=4, label='Kp')
        ax2.plot(epochs, self.ki_values, 'g-', linewidth=2, marker='s', markersize=4, label='Ki')
        ax2.plot(epochs, self.kd_values, 'b-', linewidth=2, marker='^', markersize=4, label='Kd')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Parameter Value', fontsize=12)
        ax2.set_title('PID Parameter Evolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def clear(self) -> None:
        """Clear all recorded data."""
        self.epochs.clear()
        self.mse_values.clear()
        self.kp_values.clear()
        self.ki_values.clear()
        self.kd_values.clear()

    def get_summary(self) -> dict:
        """Get summary statistics of training.

        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            'total_epochs': len(self.epochs),
            'final_mse': self.mse_values[-1] if self.mse_values else None,
            'best_mse': min(self.mse_values) if self.mse_values else None,
            'average_mse': sum(self.mse_values) / len(self.mse_values) if self.mse_values else None,
        }
        
        if self.kp_values:
            summary.update({
                'final_kp': self.kp_values[-1],
                'final_ki': self.ki_values[-1],
                'final_kd': self.kd_values[-1],
            })
        
        return summary


def plot_simulation_results(
    time: jnp.ndarray,
    reference: jnp.ndarray,
    output: jnp.ndarray,
    control: jnp.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Simulation Results",
    save_path: Optional[str] = None,
) -> "Figure":
    """Plot simulation results with output tracking and control signal.

    Args:
        time: Time array.
        reference: Reference/setpoint values.
        output: Plant output values.
        control: Control signal values.
        figsize: Figure size (width, height).
        title: Overall plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure object.
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top plot: Output tracking
    ax1.plot(time, reference, 'r--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(time, output, 'b-', linewidth=2, label='Output')
    ax1.set_ylabel('Output', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Control signal
    ax2.plot(time, control, 'g-', linewidth=2, label='Control Signal')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Control', fontsize=12)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
