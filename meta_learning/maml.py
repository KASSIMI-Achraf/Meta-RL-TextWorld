"""
MAML: Model-Agnostic Meta-Learning

Implementation of MAML (Finn et al., 2017) for meta-reinforcement learning
in text-based games.

Key concepts:
- Learn initial parameters that can quickly adapt to new tasks
- Inner loop: task-specific gradient updates
- Outer loop: meta-update to improve initial parameters
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from agents.base_agent import BaseAgent
from envs.textworld_env import TextWorldEnv
from .inner_loop import InnerLoop, collect_trajectories, Trajectory
from .outer_loop import OuterLoop


class MAML:
    """
    Model-Agnostic Meta-Learning for text-based RL.
    
    MAML learns an initialization of the agent's parameters such that
    a few gradient steps on a new task leads to good performance.
    
    Attributes:
        agent: The meta-learner agent
        inner_loop: Task adaptation component
        outer_loop: Meta-optimization component
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        meta_batch_size: int = 4,
        num_adaptation_episodes: int = 5,
        num_meta_episodes: int = 10,
        first_order: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize MAML.
        
        Args:
            agent: The agent to meta-train
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            num_inner_steps: Number of gradient steps for adaptation
            meta_batch_size: Tasks per meta-batch
            num_adaptation_episodes: Episodes for adaptation
            num_meta_episodes: Episodes for meta-gradient
            first_order: Use first-order approximation (FOMAML)
            gamma: Discount factor
            gae_lambda: GAE lambda
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping norm
            device: Device to run on
        """
        self.agent = agent
        self.device = device
        self.first_order = first_order
        
        # Initialize inner loop
        self.inner_loop = InnerLoop(
            inner_lr=inner_lr,
            num_steps=num_inner_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_coef=value_coef
        )
        
        # Initialize outer loop
        self.outer_loop = OuterLoop(
            inner_loop=self.inner_loop,
            outer_lr=outer_lr,
            meta_batch_size=meta_batch_size,
            num_adaptation_episodes=num_adaptation_episodes,
            num_meta_episodes=num_meta_episodes,
            first_order=first_order,
            max_grad_norm=max_grad_norm
        )
        
        # Store config
        self.config = {
            "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "num_inner_steps": num_inner_steps,
            "meta_batch_size": meta_batch_size,
            "num_adaptation_episodes": num_adaptation_episodes,
            "num_meta_episodes": num_meta_episodes,
            "first_order": first_order,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "max_grad_norm": max_grad_norm,
        }
    
    def meta_train(
        self,
        train_envs: List[TextWorldEnv],
        val_envs: List[TextWorldEnv],
        num_iterations: int = 1000,
        val_every: int = 50,
        save_every: int = 100,
        save_dir: str = "checkpoints",
        logger: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """
        Run meta-training.
        
        Args:
            train_envs: Training task environments
            val_envs: Validation task environments
            num_iterations: Number of meta-update iterations
            val_every: Validate every N iterations
            save_every: Save checkpoint every N iterations
            save_dir: Checkpoint directory
            logger: Optional logger
            
        Returns:
            Training history
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        self.agent.to(self.device)
        
        history = self.outer_loop.meta_train(
            agent=self.agent,
            train_envs=train_envs,
            val_envs=val_envs,
            num_iterations=num_iterations,
            val_every=val_every,
            save_every=save_every,
            save_dir=save_dir,
            logger=logger
        )
        
        return history
    
    def adapt(
        self,
        env: TextWorldEnv,
        num_episodes: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Adapt to a new task.
        
        Args:
            env: Task environment
            num_episodes: Override number of adaptation episodes
            
        Returns:
            adapted_params: Adapted parameters
            metrics: Adaptation metrics
        """
        if num_episodes is None:
            num_episodes = self.outer_loop.num_adaptation_episodes
        
        return self.inner_loop.adapt(self.agent, env, num_episodes)
    
    def evaluate(
        self,
        env: TextWorldEnv,
        num_episodes: int = 10,
        adaptation_episodes: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate adaptation on a task.
        
        Args:
            env: Task environment
            num_episodes: Evaluation episodes
            adaptation_episodes: Episodes for adaptation
            
        Returns:
            Evaluation metrics
        """
        original_params = self.agent.clone_parameters()
        
        # Adapt
        self.adapt(env, adaptation_episodes)
        
        # Evaluate
        trajectories = collect_trajectories(
            self.agent, env, num_episodes, deterministic=True
        )
        
        # Restore
        self.agent.load_parameters(original_params)
        
        return {
            "mean_reward": np.mean([t.total_reward() for t in trajectories]),
            "std_reward": np.std([t.total_reward() for t in trajectories]),
            "success_rate": np.mean([t.is_success() for t in trajectories]),
        }
    
    def save(self, path: str):
        """Save MAML state."""
        torch.save({
            "agent_state": self.agent.state_dict(),
            "config": self.config,
        }, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        agent: BaseAgent,
        device: str = "cpu"
    ) -> "MAML":
        """
        Load MAML from checkpoint.
        
        Args:
            path: Checkpoint path
            agent: Agent instance (architecture must match)
            device: Device to load to
            
        Returns:
            Loaded MAML instance
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        maml = cls(agent=agent, device=device, **config)
        maml.agent.load_state_dict(checkpoint["agent_state"])
        
        return maml
