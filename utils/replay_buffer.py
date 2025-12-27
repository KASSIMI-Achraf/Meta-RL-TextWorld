"""
Replay Buffer

Storage for trajectories and experience data.
Used for batch updates and experience replay.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from collections import deque
import random


@dataclass
class Trajectory:
    """
    Stores a single episode trajectory.
    
    This is a simpler version focused on storage rather than
    the training-specific version in inner_loop.py.
    """
    observations: List[Dict[str, str]] = field(default_factory=list)
    admissible_commands: List[List[str]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def total_reward(self) -> float:
        return sum(self.rewards)
    
    def is_success(self) -> bool:
        if len(self.infos) > 0:
            return self.infos[-1].get("won", False)
        return False
    
    def add_step(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        action: int,
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ):
        """Add a single step to the trajectory."""
        self.observations.append(observation)
        self.admissible_commands.append(admissible_commands)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
    
    def get_step(self, idx: int) -> Tuple:
        """Get a single step by index."""
        return (
            self.observations[idx],
            self.admissible_commands[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.infos[idx]
        )


@dataclass
class TransitionBatch:
    """Batch of transitions for training."""
    observations: List[Dict[str, str]]
    admissible_commands: List[List[str]]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_observations: List[Dict[str, str]]
    next_admissible_commands: List[List[str]]


class ReplayBuffer:
    """
    Replay buffer for storing and sampling trajectories.
    
    Supports both trajectory-level and transition-level sampling.
    """
    
    def __init__(
        self,
        max_trajectories: int = 1000,
        max_transitions: Optional[int] = None
    ):
        """
        Initialize the replay buffer.
        
        Args:
            max_trajectories: Maximum number of trajectories to store
            max_transitions: Maximum number of transitions (if set, overrides trajectory limit)
        """
        self.max_trajectories = max_trajectories
        self.max_transitions = max_transitions
        
        self.trajectories: deque = deque(maxlen=max_trajectories)
        self._total_transitions = 0
    
    def add_trajectory(self, trajectory: Trajectory):
        """
        Add a trajectory to the buffer.
        
        Args:
            trajectory: Trajectory to add
        """
        if self.max_transitions is not None:
            # Remove old trajectories if transition limit exceeded
            while (self._total_transitions + len(trajectory) > self.max_transitions 
                   and len(self.trajectories) > 0):
                old_traj = self.trajectories.popleft()
                self._total_transitions -= len(old_traj)
        
        self.trajectories.append(trajectory)
        self._total_transitions += len(trajectory)
    
    def sample_trajectories(self, num_trajectories: int) -> List[Trajectory]:
        """
        Sample random trajectories from the buffer.
        
        Args:
            num_trajectories: Number of trajectories to sample
            
        Returns:
            List of sampled trajectories
        """
        if len(self.trajectories) < num_trajectories:
            return list(self.trajectories)
        
        return random.sample(list(self.trajectories), num_trajectories)
    
    def sample_transitions(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> TransitionBatch:
        """
        Sample random transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device for tensors
            
        Returns:
            TransitionBatch with sampled data
        """
        # Collect all transitions
        all_transitions = []
        for traj in self.trajectories:
            for i in range(len(traj) - 1):  # Exclude last step (no next obs)
                all_transitions.append((traj, i))
        
        if len(all_transitions) < batch_size:
            batch_size = len(all_transitions)
        
        sampled = random.sample(all_transitions, batch_size)
        
        # Build batch
        observations = []
        admissible_commands = []
        actions = []
        rewards = []
        dones = []
        next_observations = []
        next_admissible_commands = []
        
        for traj, idx in sampled:
            observations.append(traj.observations[idx])
            admissible_commands.append(traj.admissible_commands[idx])
            actions.append(traj.actions[idx])
            rewards.append(traj.rewards[idx])
            dones.append(traj.dones[idx])
            next_observations.append(traj.observations[idx + 1])
            next_admissible_commands.append(traj.admissible_commands[idx + 1])
        
        return TransitionBatch(
            observations=observations,
            admissible_commands=admissible_commands,
            actions=torch.tensor(actions, device=device),
            rewards=torch.tensor(rewards, device=device, dtype=torch.float),
            dones=torch.tensor(dones, device=device, dtype=torch.float),
            next_observations=next_observations,
            next_admissible_commands=next_admissible_commands
        )
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)
    
    @property
    def num_transitions(self) -> int:
        """Return total number of transitions."""
        return self._total_transitions
    
    def clear(self):
        """Clear the buffer."""
        self.trajectories.clear()
        self._total_transitions = 0


class TaskReplayBuffer:
    """
    Replay buffer organized by task.
    
    Maintains separate buffers for each task, useful for
    meta-learning where we need task-specific data.
    """
    
    def __init__(self, max_trajectories_per_task: int = 100):
        """
        Initialize the task replay buffer.
        
        Args:
            max_trajectories_per_task: Maximum trajectories per task
        """
        self.max_per_task = max_trajectories_per_task
        self.task_buffers: Dict[str, ReplayBuffer] = {}
    
    def add_trajectory(self, task_id: str, trajectory: Trajectory):
        """Add trajectory for a specific task."""
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = ReplayBuffer(
                max_trajectories=self.max_per_task
            )
        self.task_buffers[task_id].add_trajectory(trajectory)
    
    def sample_from_task(
        self,
        task_id: str,
        num_trajectories: int
    ) -> List[Trajectory]:
        """Sample trajectories from a specific task."""
        if task_id not in self.task_buffers:
            return []
        return self.task_buffers[task_id].sample_trajectories(num_trajectories)
    
    def get_task_ids(self) -> List[str]:
        """Get all task IDs with stored data."""
        return list(self.task_buffers.keys())
    
    def clear_task(self, task_id: str):
        """Clear data for a specific task."""
        if task_id in self.task_buffers:
            self.task_buffers[task_id].clear()
    
    def clear_all(self):
        """Clear all data."""
        self.task_buffers.clear()
