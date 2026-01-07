"""
Meta-RL Agent

A text-based RL agent designed for meta-learning, featuring:
- DistilBERT text encoder for observations and commands
- Actor-critic architecture with separate policy and value heads
- Support for both MAML (functional forward) and RL² (hidden state)
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import BaseAgent
from .text_encoder import DistilBERTEncoder, CommandScorer


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Uses command scoring with learned value head.
    """
    
    def __init__(
        self,
        hidden_size: int,
        policy_hidden_sizes: List[int] = [256, 128]
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Command scorer for action probabilities
        self.command_scorer = CommandScorer(hidden_size, temperature=1.0)
        
        # MLP for additional policy features
        layers = []
        prev_size = hidden_size
        for h_size in policy_hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = h_size
        
        self.policy_features = nn.Sequential(*layers)
        self.policy_out = nn.Linear(prev_size, hidden_size)
    
    def forward(
        self,
        obs_encoding: torch.Tensor,
        cmd_encodings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action probabilities.
        
        Args:
            obs_encoding: Observation encoding (hidden_size,) or (batch, hidden)
            cmd_encodings: Command encodings (num_cmds, hidden) or (batch, num_cmds, hidden)
            
        Returns:
            Action probabilities
        """
        # Transform observation through policy features
        policy_features = self.policy_features(obs_encoding)
        transformed_obs = self.policy_out(policy_features)
        
        # Score commands
        probs = self.command_scorer(transformed_obs, cmd_encodings)
        
        return probs


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value.
    """
    
    def __init__(
        self,
        hidden_size: int,
        value_hidden_sizes: List[int] = [256, 128]
    ):
        super().__init__()
        
        layers = []
        prev_size = hidden_size
        for h_size in value_hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = h_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.value_net = nn.Sequential(*layers)
    
    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.
        
        Args:
            encoding: State encoding (hidden_size,) or (batch, hidden)
            
        Returns:
            Value estimate
        """
        return self.value_net(encoding).squeeze(-1)


class MetaRLAgent(BaseAgent):
    """
    Meta-learning compatible agent for text-based games.
    
    Combines DistilBERT encoder with actor-critic architecture.
    Supports both MAML (via functional forward with `higher`) and
    RL² (via hidden state propagation).
    
    Attributes:
        encoder: DistilBERT text encoder
        policy: Policy network
        value: Value network
        hidden_size: Size of encoding vectors
    """
    
    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        policy_hidden_sizes: List[int] = [256, 128],
        value_hidden_sizes: List[int] = [256, 128],
        hidden_size: int = 312,  # TinyBERT default
        device: str = "cpu"
    ):
        """
        Initialize the meta-RL agent.
        
        Args:
            encoder_config: Configuration for text encoder (TinyBERT default)
            policy_hidden_sizes: Hidden layer sizes for policy network
            value_hidden_sizes: Hidden layer sizes for value network
            hidden_size: Base hidden size (312 for TinyBERT)
            device: Device to run on
        """
        super().__init__()
        
        self._device = device
        self.hidden_size = hidden_size
        self._policy_hidden_sizes = policy_hidden_sizes
        self._value_hidden_sizes = value_hidden_sizes
        
        # Default encoder config - TinyBERT
        if encoder_config is None:
            encoder_config = {
                "model_name": "huawei-noah/TinyBERT_General_4L_312D",
                "freeze_layers": 2,
                "max_length": 512,
                "hidden_size": 312,
                "output_size": hidden_size,
            }
        self._encoder_config = encoder_config
        
        # Initialize encoder
        self.encoder = DistilBERTEncoder(**encoder_config, device=device)
        
        # Get actual hidden size from encoder
        actual_hidden = self.encoder.hidden_size
        
        # Initialize policy and value networks
        self.policy = PolicyNetwork(actual_hidden, policy_hidden_sizes)
        self.value = ValueNetwork(actual_hidden, value_hidden_sizes)
        
        # For RL²: optional hidden state
        self._hidden_state = None
        self._use_hidden = False
    
    def encode_observation(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        """
        Encode observation and commands.
        
        Args:
            observation: Dictionary with text fields
            admissible_commands: List of command strings
            
        Returns:
            Observation encoding
        """
        # Build full observation text
        obs_text = observation.get("text", "")
        if not obs_text:
            obs_text = (
                f"Description: {observation.get('description', '')}\n"
                f"Inventory: {observation.get('inventory', '')}\n"
                f"Feedback: {observation.get('feedback', '')}"
            )
        
        return self.encoder.encode(obs_text)
    
    def _encode_commands(self, commands: List[str]) -> torch.Tensor:
        """Encode list of commands."""
        device = next(self.parameters()).device
        if len(commands) == 0:
            return torch.zeros(1, self.encoder.hidden_size, device=device)
        return self.encoder.encode_batch(commands)
    
    def select_action(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given observation and admissible commands.
        
        Args:
            observation: Text observation dictionary
            admissible_commands: List of admissible command strings
            deterministic: If True, select argmax action
            
        Returns:
            action_idx: Selected action index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        if len(admissible_commands) == 0:
            admissible_commands = ["look"]
        
        # Encode observation and commands
        obs_encoding = self.encode_observation(observation, admissible_commands)
        cmd_encodings = self._encode_commands(admissible_commands)
        
        # Get action probabilities
        probs = self.policy(obs_encoding, cmd_encodings)
        
        # Get value estimate
        value = self.value(obs_encoding)
        
        # Sample or select action
        dist = Categorical(probs)
        
        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = dist.sample().item()
        
        log_prob = dist.log_prob(torch.tensor(action_idx, device=probs.device))
        
        return action_idx, log_prob, value
    
    def evaluate_actions(
        self,
        observations: List[Dict[str, str]],
        admissible_commands_list: List[List[str]],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for computing policy gradient losses.
        
        Args:
            observations: List of observation dictionaries
            admissible_commands_list: List of admissible command lists
            actions: Tensor of action indices
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropies: Policy entropies
        """
        batch_size = len(observations)
        device = actions.device
        
        log_probs = []
        values = []
        entropies = []
        
        for i in range(batch_size):
            obs = observations[i]
            cmds = admissible_commands_list[i]
            if len(cmds) == 0:
                cmds = ["look"]
            
            # Encode
            obs_encoding = self.encode_observation(obs, cmds)
            cmd_encodings = self._encode_commands(cmds)
            
            # Get probabilities and value
            probs = self.policy(obs_encoding, cmd_encodings)
            value = self.value(obs_encoding)
            
            # Compute log prob and entropy
            dist = Categorical(probs)
            action = actions[i]
            
            # Clamp action to valid range
            action = action.clamp(0, len(cmds) - 1)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
        
        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies)
        )
    
    def get_value(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            observation: Text observation
            admissible_commands: Admissible commands
            
        Returns:
            Value estimate
        """
        obs_encoding = self.encode_observation(observation, admissible_commands)
        return self.value(obs_encoding)
    
    def get_action_probs(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        """
        Get action probabilities without sampling.
        
        Args:
            observation: Text observation
            admissible_commands: Admissible commands
            
        Returns:
            Action probability distribution
        """
        if len(admissible_commands) == 0:
            admissible_commands = ["look"]
        
        obs_encoding = self.encode_observation(observation, admissible_commands)
        cmd_encodings = self._encode_commands(admissible_commands)
        
        return self.policy(obs_encoding, cmd_encodings)
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "encoder_config": self._encoder_config,
            "policy_hidden_sizes": self._policy_hidden_sizes,
            "value_hidden_sizes": self._value_hidden_sizes,
            "hidden_size": self.hidden_size,
            "device": self._device,
        }
    
    # RL² support: hidden state management
    def set_use_hidden(self, use_hidden: bool):
        """Enable/disable hidden state for RL²."""
        self._use_hidden = use_hidden
        if not use_hidden:
            self._hidden_state = None
    
    def reset_hidden(self):
        """Reset hidden state (for new task in RL²)."""
        self._hidden_state = None
    
    def to(self, device):
        """Move agent to device."""
        super().to(device)
        self._device = str(device)
        return self


class RL2Agent(MetaRLAgent):
    """
    RL² agent with recurrent policy and action history tracking.
    
    Maintains hidden state across episodes within a task, allowing
    the agent to adapt through its hidden state rather than gradient updates.
    
    Key features:
    - Learned action embedding (not arbitrary normalization)
    - Timestep and episode boundary signals for time awareness
    - **Action history buffer** to detect and avoid loops
    - **2-layer GRU** for stronger memory capacity
    - **Action repeat signal** to penalize repetitive behavior
    """
    
    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        policy_hidden_sizes: List[int] = [256, 128],
        value_hidden_sizes: List[int] = [256, 128],
        hidden_size: int = 312,  # TinyBERT default
        rnn_hidden_size: int = 256,
        action_embed_dim: int = 16,
        max_actions: int = 50,
        max_timesteps: int = 100,
        action_history_len: int = 10,  # Track last N actions for loop detection
        device: str = "cpu"
    ):
        """
        Initialize RL² agent with recurrent components and action history.
        
        Args:
            encoder_config: Text encoder config (TinyBERT default)
            policy_hidden_sizes: Policy MLP hidden sizes
            value_hidden_sizes: Value MLP hidden sizes
            hidden_size: Base hidden size (312 for TinyBERT)
            rnn_hidden_size: RNN hidden state size
            action_embed_dim: Dimension of action embedding
            max_actions: Maximum number of actions (for embedding table)
            max_timesteps: Maximum timesteps per episode (for timestep embedding)
            action_history_len: Length of action history buffer for loop detection
            device: Device to run on
        """
        super().__init__(
            encoder_config=encoder_config,
            policy_hidden_sizes=policy_hidden_sizes,
            value_hidden_sizes=value_hidden_sizes,
            hidden_size=hidden_size,
            device=device
        )
        
        self.rnn_hidden_size = rnn_hidden_size
        self.action_embed_dim = action_embed_dim
        self.max_actions = max_actions
        self.max_timesteps = max_timesteps
        self.action_history_len = action_history_len
        
        self.obs_projection = nn.Linear(hidden_size, rnn_hidden_size)
        
        # Learned action embedding (replaces prev_action / 20.0)
        self.action_embedding = nn.Embedding(max_actions, action_embed_dim)
        
        # Timestep embedding for time awareness
        self.timestep_embedding = nn.Embedding(max_timesteps, action_embed_dim)
        
        # Action history embedding - summarizes recent actions
        # Each position in history gets its own embedding to preserve order
        self.history_position_embedding = nn.Embedding(action_history_len, action_embed_dim // 2)
        self.history_combiner = nn.Linear(action_history_len * (action_embed_dim + action_embed_dim // 2), action_embed_dim)
        
        # RNN input: projected_obs + action_embed + reward + timestep_embed + done_flag + action_history + state_revisit_signal
        rnn_input_size = rnn_hidden_size + action_embed_dim + 1 + action_embed_dim + 1 + action_embed_dim + 1
        
        
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,  
            batch_first=True,
            dropout=0.1  
        )
        
]
        self.policy_adapter = nn.Linear(rnn_hidden_size, hidden_size)
        self.value_adapter = nn.Linear(rnn_hidden_size, hidden_size)
        
        self._hidden_state = None
        self._prev_action = 0
        self._prev_reward = 0.0
        self._prev_done = False
        self._timestep = 0
        self._action_history = []
        self._state_hashes = set()  
    
    def forward_with_hidden(
        self,
        obs_encoding: torch.Tensor,
        prev_action: int,
        prev_reward: float,
        timestep: int,
        prev_done: bool = False,
        hidden: Optional[torch.Tensor] = None,
        action_history: Optional[List[int]] = None,
        state_hash: Optional[str] = None 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hidden state, action history, and state revisit detection.
        
        Args:
            obs_encoding: Encoded observation
            prev_action: Previous action index
            prev_reward: Previous reward
            timestep: Current timestep within episode
            prev_done: Whether previous step was terminal (episode boundary)
            hidden: Previous hidden state
            action_history: List of recent actions
            state_hash: Hash of current observation for loop detection
            
        Returns:
            output: RNN output
            new_hidden: Updated hidden state
        """
        device = obs_encoding.device
        
        # Clamp action to valid embedding range
        prev_action_clamped = min(prev_action, self.max_actions - 1)
        prev_action_embed = self.action_embedding(
            torch.tensor([prev_action_clamped], device=device)
        )
        
        # Timestep embedding for time awareness
        timestep_clamped = min(timestep, self.max_timesteps - 1)
        timestep_embed = self.timestep_embedding(
            torch.tensor([timestep_clamped], device=device)
        )
        
       
        if action_history is None:
            action_history = self._action_history
        
        # Pad history to fixed length
        padded_history = [0] * self.action_history_len
        for i, act in enumerate(action_history[-self.action_history_len:]):
            padded_history[i] = min(act, self.max_actions - 1)
        
        # Compute history embedding: each action + its position
        history_features = []
        for pos, act in enumerate(padded_history):
            act_embed = self.action_embedding(torch.tensor([act], device=device))
            pos_embed = self.history_position_embedding(torch.tensor([pos], device=device))
            history_features.append(torch.cat([act_embed, pos_embed], dim=-1))
        
        history_concat = torch.cat(history_features, dim=-1)
        history_embed = self.history_combiner(history_concat)
        
        state_revisit = 1.0 if state_hash in self._state_hashes else 0.0
        state_revisit_signal = torch.tensor([[state_revisit]], device=device)
        
        prev_reward_t = torch.tensor([[prev_reward]], device=device)
        prev_done_t = torch.tensor([[float(prev_done)]], device=device)
        
        if obs_encoding.dim() == 1:
            obs_encoding = obs_encoding.unsqueeze(0)
        
        # Project observation to smaller dimension
        obs_projected = self.obs_projection(obs_encoding)
        
        # Concatenate all features including history and state revisit signal
        rnn_input = torch.cat([
            obs_projected,
            prev_action_embed,
            prev_reward_t,
            timestep_embed,
            prev_done_t,
            history_embed,
            state_revisit_signal  
        ], dim=-1)
        rnn_input = rnn_input.unsqueeze(1) 
        
        if hidden is None:
            hidden = torch.zeros(2, 1, self.rnn_hidden_size, device=device)
        
        output, new_hidden = self.rnn(rnn_input, hidden)
        
        return output.squeeze(1), new_hidden
    
    def select_action(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action with hidden state update and state tracking for loop detection.
        """
        if len(admissible_commands) == 0:
            admissible_commands = ["look"]
        
        # Compute state hash for revisit detection
        # Hash based on description + inventory (the core state identifiers)
        state_text = f"{observation.get('description', '')}{observation.get('inventory', '')}"
        state_hash = hash(state_text)
        
        obs_encoding = self.encode_observation(observation, admissible_commands)
        cmd_encodings = self._encode_commands(admissible_commands)
        
        rnn_output, self._hidden_state = self.forward_with_hidden(
            obs_encoding,
            self._prev_action,
            self._prev_reward,
            self._timestep,
            self._prev_done,
            self._hidden_state,
            self._action_history,
            state_hash  # Pass state hash for revisit detection
        )
        
        policy_input = self.policy_adapter(rnn_output)
        value_input = self.value_adapter(rnn_output)
        
        if cmd_encodings.dim() == 2:
            cmd_encodings = cmd_encodings.unsqueeze(0) 
        
        probs = self.policy.command_scorer(policy_input, cmd_encodings)
        if probs.dim() == 2:
            probs = probs.squeeze(0)
        
        value = self.value(value_input)
        
        dist = Categorical(probs)
        
        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = dist.sample().item()
        
        log_prob = dist.log_prob(torch.tensor(action_idx, device=probs.device))
        
        # Update state for next step
        self._prev_action = action_idx
        self._timestep += 1
        
        # Update action history buffer (keep last N actions)
        self._action_history.append(action_idx)
        if len(self._action_history) > self.action_history_len:
            self._action_history = self._action_history[-self.action_history_len:]
        
        # Track visited states for this episode
        self._state_hashes.add(state_hash)
        
        return action_idx, log_prob, value
    
    def update_prev_reward(self, reward: float, done: bool = False):
        """Update previous reward and done flag for next step."""
        self._prev_reward = reward
        self._prev_done = done
        if done:
            self._timestep = 0
            # Clear state hashes on episode end (new episode = fresh state tracking)
            self._state_hashes.clear()
    
    def reset_hidden(self):
        """Reset hidden state and all tracking (for new task in RL²)."""
        self._hidden_state = None
        self._prev_action = 0
        self._prev_reward = 0.0
        self._prev_done = False
        self._timestep = 0
        self._action_history = []
        self._state_hashes = set()  # Clear visited states for new task
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["rnn_hidden_size"] = self.rnn_hidden_size
        config["action_embed_dim"] = self.action_embed_dim
        config["max_actions"] = self.max_actions
        config["max_timesteps"] = self.max_timesteps
        config["action_history_len"] = self.action_history_len  # NEW
        return config
