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
        hidden_size: int = 768,
        device: str = "cpu"
    ):
        """
        Initialize the meta-RL agent.
        
        Args:
            encoder_config: Configuration for DistilBERT encoder
            policy_hidden_sizes: Hidden layer sizes for policy network
            value_hidden_sizes: Hidden layer sizes for value network
            hidden_size: Base hidden size (from encoder)
            device: Device to run on
        """
        super().__init__()
        
        self._device = device
        self.hidden_size = hidden_size
        self._policy_hidden_sizes = policy_hidden_sizes
        self._value_hidden_sizes = value_hidden_sizes
        
        # Default encoder config
        if encoder_config is None:
            encoder_config = {
                "model_name": "distilbert-base-uncased",
                "freeze_layers": 4,
                "max_length": 512,
                "hidden_size": 768,
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
        if len(commands) == 0:
            return torch.zeros(1, self.encoder.hidden_size, device=self._device)
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
    RL² agent with recurrent policy.
    
    Maintains hidden state across episodes within a task, allowing
    the agent to adapt through its hidden state rather than gradient updates.
    """
    
    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        policy_hidden_sizes: List[int] = [256, 128],
        value_hidden_sizes: List[int] = [256, 128],
        hidden_size: int = 768,
        rnn_hidden_size: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize RL² agent with recurrent components.
        
        Args:
            encoder_config: DistilBERT encoder config
            policy_hidden_sizes: Policy MLP hidden sizes
            value_hidden_sizes: Value MLP hidden sizes
            hidden_size: Base hidden size
            rnn_hidden_size: RNN hidden state size
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
        
        # GRU for maintaining hidden state across steps
        self.rnn = nn.GRU(
            input_size=hidden_size + 2,  # obs + prev_action + prev_reward
            hidden_size=rnn_hidden_size,
            batch_first=True
        )
        
        # Adjust policy and value networks to take RNN output
        self.policy_adapter = nn.Linear(rnn_hidden_size, hidden_size)
        self.value_adapter = nn.Linear(rnn_hidden_size, hidden_size)
        
        self._hidden_state = None
        self._prev_action = 0
        self._prev_reward = 0.0
    
    def forward_with_hidden(
        self,
        obs_encoding: torch.Tensor,
        prev_action: int,
        prev_reward: float,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hidden state.
        
        Args:
            obs_encoding: Encoded observation
            prev_action: Previous action index (normalized)
            prev_reward: Previous reward
            hidden: Previous hidden state
            
        Returns:
            output: RNN output
            new_hidden: Updated hidden state
        """
        # Concatenate observation with previous action and reward
        prev_action_t = torch.tensor([[prev_action / 20.0]], device=obs_encoding.device)
        prev_reward_t = torch.tensor([[prev_reward]], device=obs_encoding.device)
        
        if obs_encoding.dim() == 1:
            obs_encoding = obs_encoding.unsqueeze(0)
        
        rnn_input = torch.cat([obs_encoding, prev_action_t, prev_reward_t], dim=-1)
        rnn_input = rnn_input.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden if needed
        if hidden is None:
            hidden = torch.zeros(1, 1, self.rnn_hidden_size, device=obs_encoding.device)
        
        output, new_hidden = self.rnn(rnn_input, hidden)
        
        return output.squeeze(1), new_hidden.detach()
    
    def select_action(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action with hidden state update.
        """
        if len(admissible_commands) == 0:
            admissible_commands = ["look"]
        
        obs_encoding = self.encode_observation(observation, admissible_commands)
        cmd_encodings = self._encode_commands(admissible_commands)
        
        rnn_output, self._hidden_state = self.forward_with_hidden(
            obs_encoding,
            self._prev_action,
            self._prev_reward,
            self._hidden_state
        )
        
        policy_input = self.policy_adapter(rnn_output)
        value_input = self.value_adapter(rnn_output)
        
        if cmd_encodings.dim() == 2:
            cmd_encodings = cmd_encodings.unsqueeze(0) 
        
        probs = self.policy.command_scorer(policy_input, cmd_encodings)
        if probs.dim() == 2:
            probs = probs.squeeze(0)
        value = self.value.value_net(value_input)
        
        dist = Categorical(probs)
        
        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = dist.sample().item()
        
        log_prob = dist.log_prob(torch.tensor(action_idx, device=probs.device))
        
        self._prev_action = action_idx
        
        return action_idx, log_prob, value.squeeze()
    
    def update_prev_reward(self, reward: float):
        """Update previous reward for next step."""
        self._prev_reward = reward
    
    def reset_hidden(self):
        """Reset hidden state and history."""
        self._hidden_state = None
        self._prev_action = 0
        self._prev_reward = 0.0
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["rnn_hidden_size"] = self.rnn_hidden_size
        return config
