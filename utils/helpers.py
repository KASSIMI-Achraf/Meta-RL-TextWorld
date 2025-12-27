"""
Helper Utilities

Miscellaneous helper functions for:
- Configuration loading
- Random seed management
- Device selection
- General utilities
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import torch
import numpy as np


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", or "cuda:N")
        
    Returns:
        torch.device instance
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save to
    """
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """
    Format a large number with K/M/B suffixes.
    
    Args:
        n: Number to format
        
    Returns:
        Formatted string
    """
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return str(n)


def moving_average(values: list, window: int = 10) -> list:
    """
    Compute moving average of a list of values.
    
    Args:
        values: List of values
        window: Window size
        
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i + 1]))
    
    return smoothed


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def normalize_text(text: str) -> str:
    """
    Normalize text for encoding.
    
    - Removes extra whitespace
    - Converts to lowercase
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Lowercase
    text = text.lower()
    # Strip
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length (word-aware).
    
    Args:
        text: Input text
        max_length: Maximum character length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find last space before limit
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    
    if last_space > max_length // 2:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer for profiling."""
    
    def __init__(self):
        import time
        self._time = time
        self._start = None
        self._elapsed = 0
    
    def start(self):
        self._start = self._time.time()
    
    def stop(self) -> float:
        if self._start is not None:
            self._elapsed = self._time.time() - self._start
            self._start = None
        return self._elapsed
    
    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._time.time() - self._start
        return self._elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
