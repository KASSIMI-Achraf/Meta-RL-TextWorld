"""
Logger

Unified logging with TensorBoard integration and console output.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import json
import numpy as np


class Logger:
    """
    Logging utility with TensorBoard and file logging support.
    
    Provides a unified interface for:
    - TensorBoard scalar/histogram logging
    - Console output
    - File-based logging
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        log_to_file: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this experiment (auto-generated if None)
            use_tensorboard: Enable TensorBoard logging
            log_to_file: Enable file logging
            verbose: Enable console output
        """
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard
        self.log_to_file = log_to_file
        
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("Warning: TensorBoard not available")
                self.use_tensorboard = False
        
        # File logger
        self.log_file = None
        if log_to_file:
            self.log_file = open(self.log_dir / "training.log", "a")
        
        # Metric history
        self.history: Dict[str, list] = {}
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """
        Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current step/iteration
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            
            # Store in history
            if full_name not in self.history:
                self.history[full_name] = []
            self.history[full_name].append((step, value))
            
            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(full_name, value, step)
        
        # Console output
        if self.verbose:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"[Step {step}] {prefix}{metrics_str}")
        
        # File logging
        if self.log_file is not None:
            log_entry = {"step": step, "prefix": prefix, "metrics": metrics}
            self.log_file.write(json.dumps(log_entry) + "\n")
            self.log_file.flush()
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a single scalar value."""
        self.log_metrics({name: value}, step)
    
    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: int
    ):
        """Log a histogram of values."""
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text content."""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
        
        if self.log_file is not None:
            self.log_file.write(f"[{step}] {tag}: {text}\n")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        # Save to file
        hparams_path = self.log_dir / "hyperparameters.json"
        with open(hparams_path, "w") as f:
            json.dump(hparams, f, indent=2, default=str)
        
        if self.verbose:
            print("Hyperparameters:")
            for k, v in hparams.items():
                print(f"  {k}: {v}")
    
    def get_history(self, metric_name: str) -> list:
        """Get history for a specific metric."""
        return self.history.get(metric_name, [])
    
    def save_history(self, path: Optional[str] = None):
        """Save metric history to JSON file."""
        if path is None:
            path = str(self.log_dir / "history.json")
        
        # Convert to serializable format
        serializable = {
            k: [(step, float(v)) for step, v in values]
            for k, values in self.history.items()
        }
        
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    
    def close(self):
        """Close the logger and flush all buffers."""
        if self.writer is not None:
            self.writer.close()
        
        if self.log_file is not None:
            self.log_file.close()
        
        self.save_history()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConsoleLogger:
    """Simple console-only logger for debugging."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = {}
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        if self.verbose:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"[Step {step}] {prefix}{metrics_str}")
        
        for name, value in metrics.items():
            full_name = f"{prefix}{name}"
            if full_name not in self.history:
                self.history[full_name] = []
            self.history[full_name].append((step, value))
    
    def log_scalar(self, name: str, value: float, step: int):
        self.log_metrics({name: value}, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        if self.verbose:
            print("Hyperparameters:", hparams)
    
    def close(self):
        pass
