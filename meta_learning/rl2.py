"""
RL²: Learning to Reinforcement Learn

Implementation of RL² (Duan et al., 2016) for meta-reinforcement learning
in text-based games.

Key concepts:
- Agent uses recurrent policy that maintains hidden state across episodes
- Hidden state implicitly encodes task information
- No explicit gradient-based adaptation at test time
- Meta-training optimizes the recurrent policy across task distribution
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.meta_rl_agent import RL2Agent
from envs.textworld_env import TextWorldEnv
from .inner_loop import Trajectory, collect_trajectory


class RL2:
    """
    RL² (Learning to Reinforcement Learn) for text-based games.
    
    Unlike MAML, RL² uses a recurrent policy where the hidden state
    carries information about the current task. Adaptation happens
    implicitly through the hidden state rather than gradient updates.
    
    Attributes:
        agent: RL2Agent with recurrent policy
        episodes_per_trial: Number of episodes per task trial
        gamma: Discount factor
        gae_lambda: GAE lambda
    """
    
    def __init__(
        self,
        agent: RL2Agent,
        learning_rate: float = 0.001,
        episodes_per_trial: int = 10,
        meta_batch_size: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize RL².
        
        Args:
            agent: RL2Agent instance
            learning_rate: Policy optimization learning rate
            episodes_per_trial: Episodes per task trial (context length)
            meta_batch_size: Tasks per meta-batch
            gamma: Discount factor
            gae_lambda: GAE lambda
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping norm
            device: Device to run on
        """
        self.agent = agent
        self.device = device
        self.learning_rate = learning_rate
        self.episodes_per_trial = episodes_per_trial
        self.meta_batch_size = meta_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        
        # Config for saving/loading
        self.config = {
            "learning_rate": learning_rate,
            "episodes_per_trial": episodes_per_trial,
            "meta_batch_size": meta_batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "max_grad_norm": max_grad_norm,
        }
    
    def collect_trial(
        self,
        env: TextWorldEnv
    ) -> List[Trajectory]:
        """
        Collect a complete trial (multiple episodes) for a task.
        
        The agent's hidden state is maintained across episodes to allow
        it to accumulate task-specific information.
        
        Args:
            env: Task environment
            
        Returns:
            List of trajectories for the trial
        """
        self.agent.reset_hidden()
        
        trajectories = []
        
        for episode in range(self.episodes_per_trial):
            trajectory = self._collect_episode_with_hidden(env)
            trajectories.append(trajectory)
            
            # Clear cache after each episode to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return trajectories
    
    def _collect_episode_with_hidden(
        self,
        env: TextWorldEnv
    ) -> Trajectory:
        """
        Collect an episode while maintaining hidden state.
        
        Args:
            env: Task environment
            
        Returns:
            Episode trajectory
        """
        trajectory = Trajectory()
        
        obs, info = env.reset()
        done = False
        
        while not done:
            admissible_cmds = info.get("admissible_commands", ["look"])
            
            with torch.no_grad():
                action_idx, log_prob, value = self.agent.select_action(
                    obs, admissible_cmds
                )
            
            trajectory.observations.append(obs)
            trajectory.admissible_commands.append(admissible_cmds)
            trajectory.actions.append(action_idx)
            trajectory.log_probs.append(log_prob.detach())
            trajectory.values.append(value.detach())
            
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            
            trajectory.rewards.append(reward)
            trajectory.dones.append(done)
            trajectory.infos.append(info)
            
            # Pass done flag for episode boundary awareness
            self.agent.update_prev_reward(reward, done)
            
            obs = next_obs
        
        return trajectory

    def _build_obs_text(self, observation: dict) -> str:
        """Build observation text string for encoding."""
        obs_text = observation.get("text", "")
        if not obs_text:
            obs_text = (
                f"Description: {observation.get('description', '')}\n"
                f"Inventory: {observation.get('inventory', '')}\n"
                f"Feedback: {observation.get('feedback', '')}"
            )
        return obs_text
    
    def compute_trial_loss(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a complete trial by re-evaluating actions.
        
        OPTIMIZED: Batch-encodes observations upfront and caches command encodings
        to avoid expensive encoder calls inside Python loops.
        
        Args:
            trajectories: Trial trajectories
            
        Returns:
            loss: Total loss
            metrics: Loss components
        """
        device = next(self.agent.parameters()).device
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        
        # Reset hidden state for re-evaluation
        self.agent.reset_hidden()
        
        for traj in trajectories:
            if len(traj) == 0:
                continue
            
            # Compute returns and advantages from stored rewards/values
            returns, advantages = self._compute_returns_and_advantages(traj)
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            returns = returns.to(device)
            advantages = advantages.to(device)
            
            # === BATCH ENCODE ALL OBSERVATIONS UPFRONT ===
            all_obs_texts = [self._build_obs_text(obs) for obs in traj.observations]
            all_obs_encodings = self.agent.encoder.encode_batch(all_obs_texts)
            
            # === CACHE COMMAND ENCODINGS ===
            cmd_cache = {}
            for cmds in traj.admissible_commands:
                key = tuple(cmds) if cmds else ("look",)
                if key not in cmd_cache:
                    cmd_list = list(cmds) if cmds else ["look"]
                    cmd_cache[key] = self.agent._encode_commands(cmd_list)
            
            # Re-evaluate each step WITH gradients (but encodings are cached)
            log_probs_list = []
            values_list = []
            entropies_list = []
            
            # Build action history and state hash tracking as we re-evaluate
            action_history = []
            visited_states = set()
            
            for t in range(len(traj)):
                cmds = traj.admissible_commands[t]
                action = traj.actions[t]
                obs = traj.observations[t]
                
                # Compute state hash for this observation (deterministic, same as select_action)
                state_text = f"{obs.get('description', '')}{obs.get('inventory', '')}"
                state_hash = hashlib.md5(state_text.encode()).hexdigest()
                
                # Use pre-computed encodings
                obs_encoding = all_obs_encodings[t]
                cmd_key = tuple(cmds) if cmds else ("look",)
                cmd_encodings = cmd_cache[cmd_key]
                
                # Get done flag from previous step (for episode boundary signal)
                prev_done = traj.dones[t - 1] if t > 0 else False
                
                # Temporarily set agent's state hashes for forward_with_hidden
                self.agent._state_hashes = visited_states
                
                # Pass action history and state_hash for loop detection
                rnn_output, self.agent._hidden_state = self.agent.forward_with_hidden(
                    obs_encoding,
                    self.agent._prev_action,
                    self.agent._prev_reward,
                    self.agent._timestep,
                    prev_done,
                    self.agent._hidden_state,
                    action_history,
                    state_hash
                )
                
                # Track this state as visited
                visited_states.add(state_hash)
                
                policy_input = self.agent.policy_adapter(rnn_output)
                value_input = self.agent.value_adapter(rnn_output)
                
                if cmd_encodings.dim() == 2:
                    cmd_encodings = cmd_encodings.unsqueeze(0)
                
                probs = self.agent.policy.command_scorer(policy_input, cmd_encodings)
                if probs.dim() == 2:
                    probs = probs.squeeze(0)
                
                # Use proper ValueNetwork.forward()
                value = self.agent.value(value_input)
                
                dist = torch.distributions.Categorical(probs)
                action_tensor = torch.tensor(action, device=device)
                cmd_count = len(cmds) if cmds else 1
                action_tensor = action_tensor.clamp(0, cmd_count - 1)
                
                log_prob = dist.log_prob(action_tensor)
                entropy = dist.entropy()
                
                log_probs_list.append(log_prob)
                values_list.append(value)
                entropies_list.append(entropy)
                
                # Update prev action/reward/timestep for next step
                self.agent._prev_action = action
                self.agent._timestep += 1
                
                # Update action history (mirroring select_action behavior)
                action_history.append(action)
                if len(action_history) > self.agent.action_history_len:
                    action_history = action_history[-self.agent.action_history_len:]
                
                if t < len(traj.rewards):
                    self.agent._prev_reward = traj.rewards[t]
                    # Reset timestep on episode boundary
                    if traj.dones[t]:
                        self.agent._timestep = 0
                        action_history = []  # Reset history on episode boundary
                        visited_states = set()  # Reset on episode boundary
            
            log_probs = torch.stack(log_probs_list)
            values = torch.stack(values_list)
            entropies = torch.stack(entropies_list)
            
            # Policy loss
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns.detach())
            
            # Entropy bonus
            entropy = entropies.mean()
            
            total_policy_loss += policy_loss * len(traj)
            total_value_loss += value_loss * len(traj)
            total_entropy += entropy * len(traj)
            total_steps += len(traj)
        
        if total_steps == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        avg_policy_loss = total_policy_loss / total_steps
        avg_value_loss = total_value_loss / total_steps
        avg_entropy = total_entropy / total_steps
        
        loss = avg_policy_loss + self.value_coef * avg_value_loss - self.entropy_coef * avg_entropy
        
        metrics = {
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "entropy": avg_entropy.item(),
        }
        
        return loss, metrics
    
    def _compute_returns_and_advantages(
        self,
        trajectory: Trajectory,
        last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages."""
        T = len(trajectory.rewards)
        returns = torch.zeros(T)
        advantages = torch.zeros(T)
        
        values = [v.item() if torch.is_tensor(v) else v for v in trajectory.values]
        
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = (trajectory.rewards[t] + 
                     self.gamma * next_value * (1 - trajectory.dones[t]) - 
                     values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - trajectory.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def meta_update(
        self,
        task_envs: List[TextWorldEnv]
    ) -> Dict[str, float]:
        """
        Perform one meta-update step.
        
        Args:
            task_envs: List of task environments
            
        Returns:
            Metrics dictionary
        """
        if len(task_envs) > self.meta_batch_size:
            indices = np.random.choice(
                len(task_envs), self.meta_batch_size, replace=False
            )
            batch_envs = [task_envs[i] for i in indices]
        else:
            batch_envs = task_envs
        
        total_loss = 0.0
        all_rewards = []
        all_success = []
        
        for env in batch_envs:
            trajectories = self.collect_trial(env)
            
            loss, _ = self.compute_trial_loss(trajectories)
            total_loss += loss
            
            for traj in trajectories:
                all_rewards.append(traj.total_reward())
                all_success.append(traj.is_success())
            
            # Cleanup after each task to prevent memory accumulation
            del trajectories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(batch_envs)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Clear CUDA cache after update (this was unreachable before!)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "loss": avg_loss.item(),
            "mean_reward": np.mean(all_rewards),
            "success_rate": np.mean(all_success),
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
        
        history = {
            "loss": [],
            "train_reward": [],
            "train_success": [],
            "val_reward": [],
            "val_success": [],
        }
        
        best_val_reward = float("-inf")
        
        # Progress bar
        pbar = tqdm(range(num_iterations), desc="Meta-Training")
        
        for iteration in pbar:
            metrics = self.meta_update(train_envs)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{metrics['mean_reward']:.3f}",
                "win": f"{metrics['success_rate']:.1%}"
            })
            
            history["loss"].append(metrics["loss"])
            history["train_reward"].append(metrics["mean_reward"])
            history["train_success"].append(metrics["success_rate"])
            
            if logger:
                logger.log_metrics(metrics, iteration)
            
            # Validate
            if (iteration + 1) % val_every == 0:
                val_metrics = self.evaluate(val_envs)
                history["val_reward"].append(val_metrics["mean_reward"])
                history["val_success"].append(val_metrics["success_rate"])
                
                if logger:
                    logger.log_metrics(
                        {"val_" + k: v for k, v in val_metrics.items()}, 
                        iteration
                    )
                
                if val_metrics["mean_reward"] > best_val_reward:
                    best_val_reward = val_metrics["mean_reward"]
                    self.save(f"{save_dir}/best_model.pt")
            
            # Save checkpoint
            if (iteration + 1) % save_every == 0:
                self.save(f"{save_dir}/checkpoint_{iteration + 1}.pt")
        
        return history
    
    def evaluate(
        self,
        test_envs: List[TextWorldEnv],
        num_trials: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate on test tasks.
        
        Args:
            test_envs: Test task environments
            num_trials: Number of trials per task
            
        Returns:
            Evaluation metrics
        """
        all_rewards = []
        all_success = []
        
        for env in test_envs:
            for _ in range(num_trials):
                trajectories = self.collect_trial(env)
                
                # Use later episodes (after adaptation)
                for traj in trajectories[len(trajectories)//2:]:
                    all_rewards.append(traj.total_reward())
                    all_success.append(traj.is_success())
        
        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "success_rate": np.mean(all_success),
        }
    
    def adapt(
        self,
        env: TextWorldEnv,
        num_episodes: int = 5
    ) -> List[Trajectory]:
        """
        Adapt to a new task by running episodes.
        
        For RL², adaptation happens through the hidden state.
        
        Args:
            env: Task environment
            num_episodes: Number of adaptation episodes
            
        Returns:
            Adaptation trajectories
        """
        self.agent.reset_hidden()
        
        trajectories = []
        for _ in range(num_episodes):
            traj = self._collect_episode_with_hidden(env)
            trajectories.append(traj)
        
        return trajectories
    
    def save(self, path: str):
        """Save RL² state."""
        torch.save({
            "agent_state": self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        agent: RL2Agent,
        device: str = "cpu"
    ) -> "RL2":
        """Load RL² from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        rl2 = cls(agent=agent, device=device, **config)
        rl2.agent.load_state_dict(checkpoint["agent_state"])
        rl2.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        return rl2

