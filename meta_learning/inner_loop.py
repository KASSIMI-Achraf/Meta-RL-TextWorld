"""
Inner Loop: Task Adaptation

Handles trajectory collection and gradient computation for task-level
adaptation in meta-learning algorithms.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseAgent
from envs.textworld_env import TextWorldEnv


@dataclass
class Trajectory:
    """
    Stores a single episode trajectory.
    
    Attributes:
        observations: List of observation dictionaries
        admissible_commands: List of admissible command lists
        actions: List of action indices
        rewards: List of rewards
        dones: List of done flags
        log_probs: List of action log probabilities
        values: List of value estimates
        infos: List of info dictionaries
    """
    observations: List[Dict[str, str]] = field(default_factory=list)
    admissible_commands: List[List[str]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def total_reward(self) -> float:
        return sum(self.rewards)
    
    def is_success(self) -> bool:
        """Check if the episode ended in success (game won)."""
        if len(self.infos) > 0:
            return self.infos[-1].get("won", False)
        return False


@dataclass 
class TaskBatch:
    """
    Batch of trajectories from multiple tasks for meta-update.
    
    Attributes:
        support_trajectories: Trajectories for adaptation (inner loop)
        query_trajectories: Trajectories for meta-update (outer loop)
        task_ids: Identifiers for each task
    """
    support_trajectories: List[List[Trajectory]] = field(default_factory=list)
    query_trajectories: List[List[Trajectory]] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)


def collect_trajectory(
    agent: BaseAgent,
    env: TextWorldEnv,
    max_steps: Optional[int] = None,
    deterministic: bool = False
) -> Trajectory:
    """
    Collect a single episode trajectory.
    
    Args:
        agent: The agent to collect with
        env: The environment to collect in
        max_steps: Maximum steps (if None, use env default)
        deterministic: If True, select actions deterministically
        
    Returns:
        Trajectory containing the episode data
    """
    trajectory = Trajectory()
    
    obs, info = env.reset()
    done = False
    step = 0
    max_steps = max_steps or env.max_steps
    
    while not done and step < max_steps:
        admissible_cmds = info.get("admissible_commands", ["look"])
        
        # Select action
        with torch.no_grad():
            action_idx, log_prob, value = agent.select_action(
                obs, admissible_cmds, deterministic=deterministic
            )
        
        # Store transition
        trajectory.observations.append(obs)
        trajectory.admissible_commands.append(admissible_cmds)
        trajectory.actions.append(action_idx)
        trajectory.log_probs.append(log_prob)
        trajectory.values.append(value)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
        
        trajectory.rewards.append(reward)
        trajectory.dones.append(done)
        trajectory.infos.append(info)
        
        obs = next_obs
        step += 1
    
    return trajectory


def collect_trajectories(
    agent: BaseAgent,
    env: TextWorldEnv,
    num_episodes: int,
    max_steps: Optional[int] = None,
    deterministic: bool = False
) -> List[Trajectory]:
    """
    Collect multiple episode trajectories.
    
    Args:
        agent: The agent to collect with
        env: The environment to collect in
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        deterministic: If True, select actions deterministically
        
    Returns:
        List of trajectories
    """
    trajectories = []
    
    for _ in range(num_episodes):
        traj = collect_trajectory(agent, env, max_steps, deterministic)
        trajectories.append(traj)
    
    return trajectories


class InnerLoop:
    """
    Task-level adaptation loop for meta-learning.
    
    Performs gradient updates on task-specific data to adapt the agent
    to a new task quickly.
    
    Attributes:
        inner_lr: Learning rate for inner loop updates
        num_steps: Number of gradient steps
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    """
    
    def __init__(
        self,
        inner_lr: float = 0.01,
        num_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        normalize_advantages: bool = True
    ):
        """
        Initialize the inner loop.
        
        Args:
            inner_lr: Learning rate for inner loop gradient steps
            num_steps: Number of gradient steps per adaptation
            gamma: Discount factor for returns
            gae_lambda: Lambda for GAE
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            normalize_advantages: Whether to normalize advantages
        """
        self.inner_lr = inner_lr
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.normalize_advantages = normalize_advantages
    
    def compute_returns_and_advantages(
        self,
        trajectory: Trajectory,
        last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discounted returns and GAE advantages.
        
        Args:
            trajectory: Episode trajectory
            last_value: Bootstrap value for incomplete episodes
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = trajectory.rewards
        values = [v.item() if torch.is_tensor(v) else v for v in trajectory.values]
        dones = trajectory.dones
        
        T = len(rewards)
        returns = torch.zeros(T)
        advantages = torch.zeros(T)
        
        # Compute GAE
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def compute_loss(
        self,
        agent: BaseAgent,
        trajectories: List[Trajectory]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute policy gradient loss for a batch of trajectories.
        
        Args:
            agent: The agent
            trajectories: List of trajectories to compute loss on
            
        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        
        for traj in trajectories:
            if len(traj) == 0:
                continue
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(traj)
            
            if self.normalize_advantages and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Prepare batch data
            device = next(agent.parameters()).device
            actions = torch.tensor(traj.actions, device=device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            
            # Evaluate actions
            log_probs, values, entropies = agent.evaluate_actions(
                traj.observations,
                traj.admissible_commands,
                actions
            )
            
            # Policy loss (REINFORCE with baseline)
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = entropies.mean()
            
            total_policy_loss += policy_loss * len(traj)
            total_value_loss += value_loss * len(traj)
            total_entropy += entropy * len(traj)
            total_steps += len(traj)
        
        if total_steps == 0:
            return torch.tensor(0.0, requires_grad=True), {}
        
        # Average losses
        avg_policy_loss = total_policy_loss / total_steps
        avg_value_loss = total_value_loss / total_steps
        avg_entropy = total_entropy / total_steps
        
        # Total loss
        loss = avg_policy_loss + self.value_coef * avg_value_loss - self.entropy_coef * avg_entropy
        
        metrics = {
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "entropy": avg_entropy.item(),
            "total_loss": loss.item(),
        }
        
        return loss, metrics
    
    def adapt(
        self,
        agent: BaseAgent,
        env: TextWorldEnv,
        num_episodes: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Adapt the agent to a task through gradient updates.
        
        Args:
            agent: The agent to adapt
            env: The task environment
            num_episodes: Number of episodes for adaptation
            
        Returns:
            adapted_params: Updated parameters after adaptation
            metrics: Adaptation metrics
        """
        # Create optimizer for inner loop
        optimizer = torch.optim.SGD(agent.parameters(), lr=self.inner_lr)
        
        all_metrics = []
        
        for step in range(self.num_steps):
            # Collect trajectories with current policy
            trajectories = collect_trajectories(agent, env, num_episodes)
            
            # Compute loss
            loss, metrics = self.compute_loss(agent, trajectories)
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            metrics["step"] = step
            metrics["mean_reward"] = np.mean([t.total_reward() for t in trajectories])
            metrics["success_rate"] = np.mean([t.is_success() for t in trajectories])
            all_metrics.append(metrics)
        
        # Get adapted parameters
        adapted_params = agent.clone_parameters()
        
        # Aggregate metrics
        final_metrics = {
            "mean_reward": all_metrics[-1]["mean_reward"],
            "success_rate": all_metrics[-1]["success_rate"],
            "final_loss": all_metrics[-1]["total_loss"],
        }
        
        return adapted_params, final_metrics
    
    def adapt_functional(
        self,
        agent: BaseAgent,
        trajectories: List[Trajectory],
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform functional adaptation (for MAML second-order gradients).
        
        Uses the `higher` library for differentiable optimization.
        
        Args:
            agent: The agent
            trajectories: Trajectories for adaptation
            params: Current parameters
            
        Returns:
            Updated parameters (with gradient graph)
        """
        # This is a simplified version - full MAML uses `higher` library
        # for differentiable inner loop
        
        # Load parameters
        original_params = agent.clone_parameters()
        agent.load_parameters(params)
        
        # Compute loss and gradients
        loss, _ = self.compute_loss(agent, trajectories)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, agent.parameters(), create_graph=True)
        
        # Update parameters
        new_params = {}
        for (name, param), grad in zip(agent.named_parameters(), grads):
            new_params[name] = param - self.inner_lr * grad
        
        # Restore original parameters
        agent.load_parameters(original_params)
        
        return new_params
