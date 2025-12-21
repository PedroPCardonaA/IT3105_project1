"""Results exporter for saving training configuration and metrics to CSV files."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np


class ResultsExporter:
    """Export training configuration and results to CSV files."""
    
    def __init__(self):
        self.config_data = {}
        self.results_data = []
        
    def set_configuration(self, config: Dict[str, Any]):
        """Store configuration data for export.
        
        Args:
            config: Full configuration dictionary from YAML
        """
        self.config_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plant_type': config['plant']['type'],
            'controller_type': config['controller']['type'],
            'dt': config['simulation']['dt'],
            'epochs': config['training']['epochs'],
            'timesteps_per_epoch': config['training']['timesteps_per_epoch'],
            'learning_rate': config['training']['learning_rate'],
            'disturbance_min': config['simulation']['disturbance_range'][0],
            'disturbance_max': config['simulation']['disturbance_range'][1],
        }
        
        # Add plant-specific parameters
        plant_type = config['plant']['type']
        plant_params = config['plant']['params'][plant_type]
        for key, value in plant_params.items():
            # Convert lists to string representation
            if isinstance(value, list):
                value = str(value)
            self.config_data[f'plant_{key}'] = value
        
        # Add controller-specific parameters
        controller_type = config['controller']['type']
        if controller_type == 'classic':
            ctrl_params = config['controller']['classic_pid']
            for key, value in ctrl_params.items():
                self.config_data[f'controller_{key}'] = value
        elif controller_type == 'ai':
            ctrl_params = config['controller']['ai']
            for key, value in ctrl_params.items():
                if isinstance(value, list):
                    value = str(value)
                self.config_data[f'controller_{key}'] = value
    
    def add_epoch_result(
        self,
        epoch: int,
        mse: float,
        final_output: float,
        avg_control: Optional[float] = None,
        max_control: Optional[float] = None,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        **kwargs
    ):
        """Add results from a single epoch.
        
        Args:
            epoch: Epoch number
            mse: Mean squared error
            final_output: Final plant output value
            avg_control: Average control signal (optional)
            max_control: Maximum control signal (optional)
            kp: PID proportional gain (optional, for classic PID)
            ki: PID integral gain (optional, for classic PID)
            kd: PID derivative gain (optional, for classic PID)
            **kwargs: Additional metrics to record
        """
        result = {
            'epoch': epoch,
            'mse': float(mse),
            'final_output': float(final_output),
        }
        
        if avg_control is not None:
            result['avg_control'] = float(avg_control)
        if max_control is not None:
            result['max_control'] = float(max_control)
        if kp is not None:
            result['kp'] = float(kp)
        if ki is not None:
            result['ki'] = float(ki)
        if kd is not None:
            result['kd'] = float(kd)
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if value is not None:
                result[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
        
        self.results_data.append(result)
    
    def export_configuration(self, filepath: str | Path):
        """Export configuration to CSV file.
        
        Args:
            filepath: Path to save configuration CSV
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            for key, value in self.config_data.items():
                writer.writerow([key, value])
    
    def export_results(self, filepath: str | Path):
        """Export epoch-by-epoch results to CSV file.
        
        Args:
            filepath: Path to save results CSV
        """
        if not self.results_data:
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all unique keys from all results
        fieldnames = set()
        for result in self.results_data:
            fieldnames.update(result.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results_data)
    
    def export_summary(self, filepath: str | Path, summary: Dict[str, Any]):
        """Export training summary to CSV file.
        
        Args:
            filepath: Path to save summary CSV
            summary: Summary dictionary with final metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in summary.items():
                writer.writerow([key, value])
    
    def export_all(
        self,
        output_dir: str | Path,
        prefix: str = '',
        summary: Optional[Dict[str, Any]] = None
    ):
        """Export all data (configuration, results, and summary) to CSV files.
        
        Args:
            output_dir: Directory to save all CSV files
            prefix: Optional prefix for filenames
            summary: Optional summary dictionary to export
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        
        # Export configuration
        self.export_configuration(output_dir / f"{prefix}configuration.csv")
        
        # Export results
        self.export_results(output_dir / f"{prefix}training_results.csv")
        
        # Export summary if provided
        if summary is not None:
            self.export_summary(output_dir / f"{prefix}training_summary.csv", summary)
