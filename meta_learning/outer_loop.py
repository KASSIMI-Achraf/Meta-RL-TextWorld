"""
Outer Loop: Meta-Optimization

Handles the meta-update step that optimizes the initial parameters
for fast adaptation across a distribution of tasks.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from agents.base_agent import BaseAgent
from envs.textworld_env import TextWorldEnv, TextWorldBatchEnv
from .inner_loop import InnerLoop, Trajectory, TaskBatch, collect_trajectories


class OuterLoop:
    """
    Meta-optimization loop for meta-learning.
    
    Samples batches of tasks, runs inner loop adaptation on each,
    and computes meta-gradients for updating the initial parameters.
    
    Attributes:
        inner_loop: InnerLoop instance for task adaptation
        outer_lr: Learning rate for meta-optimizer
        meta_batch_size: Number of tasks per meta-batch
    """
    
    def __init__(
        self,
        inner_loop: InnerLoop,
        outer_lr: float = 0.001,
        meta_batch_size: int = 4,
        num_adaptation_episodes: int = 5,
        num_meta_episodes: int = 10,
        first_order: bool = False,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize the outer loop.
        
        Args:
            inner_loop: Inner loop instance
            outer_lr: Meta-optimizer learning rate
            meta_batch_size: Number of tasks per meta-batch
            num_adaptation_episodes: Episodes for inner loop adaptation
            num_meta_episodes: Episodes for meta-gradient computation
            first_order: Use first-order approximation (no second derivatives)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.inner_loop = inner_loop
        self.outer_lr = outer_lr
        self.meta_batch_size = meta_batch_size
        self.num_adaptation_episodes = num_adaptation_episodes
        self.num_meta_episodes = num_meta_episodes
        self.first_order = first_order
        self.max_grad_norm = max_grad_norm
    
    def sample_task_batch(
        self,
        task_envs: List[TextWorldEnv]
    ) -> List[TextWorldEnv]:
        """
        Sample a batch of tasks for meta-update.
        
        Args:
            task_envs: List of available task environments
            
        Returns:
            Sampled task environments
        """
        if len(task_envs) <= self.meta_batch_size:
            return task_envs
        
        indices = np.random.choice(
            len(task_envs),
            size=self.meta_batch_size,
            replace=False
        )
        return [task_envs[i] for i in indices]
    
    def meta_update(
        self,
        agent: BaseAgent,
        task_envs: List[TextWorldEnv],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one meta-update step.
        
        For each task in the batch:
        1. Adapt agent on support set (inner loop)
        2. Compute loss on query set with adapted parameters
        3. Accumulate meta-gradients
        
        Args:
            agent: The meta-learner agent
            task_envs: List of task environments
            optimizer: Meta-optimizer
            
        Returns:
            Metrics dictionary
        """
        # Sample task batch
        batch_envs = self.sample_task_batch(task_envs)
        
        # Store original parameters
        original_params = agent.clone_parameters()
        
        # Accumulate meta-gradients
        meta_loss = 0.0
        task_metrics = []
        
        for task_env in batch_envs:
            # Phase 1: Collect support trajectories and adapt
            support_trajectories = collect_trajectories(
                agent, task_env, self.num_adaptation_episodes
            )
            
            # Record pre-adaptation performance
            pre_adapt_reward = np.mean([t.total_reward() for t in support_trajectories])
            
            # Adapt parameters
            if self.first_order:
                # First-order: just update parameters directly
                adapted_params, adapt_metrics = self.inner_loop.adapt(
                    agent, task_env, self.num_adaptation_episodes
                )
            else:
                # Second-order: use functional adaptation (requires higher)
                for _ in range(self.inner_loop.num_steps):
                    loss, _ = self.inner_loop.compute_loss(agent, support_trajectories)
                    loss.backward()
                    
                    with torch.no_grad():
                        for param in agent.parameters():
                            if param.grad is not None:
                                param -= self.inner_loop.inner_lr * param.grad
                                param.grad.zero_()
                    
                    support_trajectories = collect_trajectories(
                        agent, task_env, self.num_adaptation_episodes
                    )
            
            # Phase 2: Collect query trajectories with adapted parameters
            query_trajectories = collect_trajectories(
                agent, task_env, self.num_meta_episodes
            )
            
            # Compute meta-loss on query set
            task_loss, loss_metrics = self.inner_loop.compute_loss(agent, query_trajectories)
            meta_loss += task_loss
            
            # Record post-adaptation performance
            post_adapt_reward = np.mean([t.total_reward() for t in query_trajectories])
            success_rate = np.mean([t.is_success() for t in query_trajectories])
            
            task_metrics.append({
                "pre_adapt_reward": pre_adapt_reward,
                "post_adapt_reward": post_adapt_reward,
                "success_rate": success_rate,
                "task_loss": task_loss.item(),
            })
            
            # Restore original parameters for next task
            agent.load_parameters(original_params)
        
        # Average meta-loss
        meta_loss = meta_loss / len(batch_envs)
        
        # Meta-update
        optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
        
        optimizer.step()
        
        # Aggregate metrics
        metrics = {
            "meta_loss": meta_loss.item(),
            "mean_pre_adapt_reward": np.mean([m["pre_adapt_reward"] for m in task_metrics]),
            "mean_post_adapt_reward": np.mean([m["post_adapt_reward"] for m in task_metrics]),
            "mean_success_rate": np.mean([m["success_rate"] for m in task_metrics]),
            "adaptation_improvement": np.mean([
                m["post_adapt_reward"] - m["pre_adapt_reward"] for m in task_metrics
            ]),
        }
        
        return metrics
    
    def meta_train(
        self,
        agent: BaseAgent,
        train_envs: List[TextWorldEnv],
        val_envs: List[TextWorldEnv],
        num_iterations: int,
        val_every: int = 50,
        save_every: int = 100,
        save_dir: str = "checkpoints",
        logger: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """
        Full meta-training loop.
        
        Args:
            agent: The agent to train
            train_envs: Training task environments
            val_envs: Validation task environments
            num_iterations: Number of meta-update iterations
            val_every: Validate every N iterations
            save_every: Save checkpoint every N iterations
            save_dir: Directory to save checkpoints
            logger: Optional logger
            
        Returns:
            Training history
        """
        # Create meta-optimizer
        optimizer = torch.optim.Adam(agent.parameters(), lr=self.outer_lr)
        
        history = {
            "meta_loss": [],
            "train_reward": [],
            "train_success": [],
            "val_reward": [],
            "val_success": [],
        }
        
        best_val_reward = float("-inf")
        
        for iteration in range(num_iterations):
            # Meta-update on training tasks
            metrics = self.meta_update(agent, train_envs, optimizer)
            
            history["meta_loss"].append(metrics["meta_loss"])
            history["train_reward"].append(metrics["mean_post_adapt_reward"])
            history["train_success"].append(metrics["mean_success_rate"])
            
            # Log
            if logger is not None:
                logger.log_metrics(metrics, iteration)
            
            # Validate
            if (iteration + 1) % val_every == 0:
                val_metrics = self.validate(agent, val_envs)
                history["val_reward"].append(val_metrics["mean_reward"])
                history["val_success"].append(val_metrics["success_rate"])
                
                if logger is not None:
                    logger.log_metrics({"val_" + k: v for k, v in val_metrics.items()}, iteration)
                
                # Save best model
                if val_metrics["mean_reward"] > best_val_reward:
                    best_val_reward = val_metrics["mean_reward"]
                    agent.save(f"{save_dir}/best_model.pt")
            
            # Save checkpoint
            if (iteration + 1) % save_every == 0:
                agent.save(f"{save_dir}/checkpoint_{iteration + 1}.pt")
        
        return history
    
    def validate(
        self,
        agent: BaseAgent,
        val_envs: List[TextWorldEnv],
        num_eval_episodes: int = 5
    ) -> Dict[str, float]:
        """
        Validate agent on held-out tasks.
        
        Args:
            agent: The agent
            val_envs: Validation environments
            num_eval_episodes: Episodes per task for evaluation
            
        Returns:
            Validation metrics
        """
        original_params = agent.clone_parameters()
        
        all_rewards = []
        all_success = []
        
        for env in val_envs:
            # Adapt to task
            self.inner_loop.adapt(agent, env, self.num_adaptation_episodes)
            
            # Evaluate
            trajectories = collect_trajectories(
                agent, env, num_eval_episodes, deterministic=True
            )
            
            rewards = [t.total_reward() for t in trajectories]
            success = [t.is_success() for t in trajectories]
            
            all_rewards.extend(rewards)
            all_success.extend(success)
            
            # Restore parameters
            agent.load_parameters(original_params)
        
        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "success_rate": np.mean(all_success),
        }
