"""
Base Agent Abstract Class

Defines the interface that all agents must implement for compatibility
with the meta-learning pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement observation encoding, action selection,
    and parameter update methods to be compatible with the meta-learning
    pipeline.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def encode_observation(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        """
        Encode a text observation into a fixed-size vector.
        
        Note: The admissible_commands parameter is provided for agents that
        may condition state encoding on available actions (e.g., for attention).
        Most implementations should encode the observation independently and
        encode commands separately. If your agent ignores admissible_commands
        in this method, that is the expected default behavior.
        
        Args:
            observation: Dictionary with 'text', 'description', 'inventory', 'feedback'
            admissible_commands: List of admissible command strings (may be ignored)
            
        Returns:
            Encoded observation tensor of shape (hidden_size,)
        """
        pass
    
    @abstractmethod
    def select_action(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given an observation.
        
        Args:
            observation: Dictionary with text observations
            admissible_commands: List of admissible command strings
            deterministic: If True, select the action with highest probability
            
        Returns:
            action_idx: Index of the selected action
            log_prob: Log probability of the selected action
            value: Estimated value of the state
        """
        pass
    
    @abstractmethod
    def evaluate_actions(
        self,
        observations: List[Dict[str, str]],
        admissible_commands_list: List[List[str]],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for a batch of observations.
        
        Used during policy gradient updates to compute losses.
        
        WARNING: This interface uses Python lists due to variable-length text
        and command sets. This forces per-sample loops and prevents vectorized
        training. For performance-critical training, implementations should:
        - Batch-encode observations upfront where possible
        - Cache command encodings for repeated command sets
        - Consider using padded tensors for fixed-batch operations
        
        Args:
            observations: List of observation dictionaries
            admissible_commands_list: List of admissible command lists (variable length)
            actions: Action indices tensor of shape (batch_size,)
            
        Returns:
            log_probs: Log probabilities of actions (batch_size,)
            values: State value estimates (batch_size,)
            entropies: Policy entropies (batch_size,)
        """
        pass
    
    @abstractmethod
    def get_value(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        """
        Get the value estimate for a state.
        
        Args:
            observation: Dictionary with text observations
            admissible_commands: List of admissible command strings
            
        Returns:
            Value estimate tensor
        """
        pass
    
    def get_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get all trainable parameters.
        
        Returns:
            Iterator over parameters
        """
        return self.parameters()
    
    def get_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Get all named trainable parameters.
        
        Returns:
            Iterator over (name, parameter) tuples
        """
        return self.named_parameters()
    
    def clone_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Create a copy of all parameters.
        
        Returns:
            Dictionary mapping parameter names to cloned tensors
        """
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def load_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Load parameters from a dictionary.
        
        Args:
            params: Dictionary mapping parameter names to tensors
        """
        state_dict = self.state_dict()
        for name, param in params.items():
            if name in state_dict:
                state_dict[name].copy_(param)
    
    def save(self, path: str):
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseAgent":
        """
        Load an agent from a file.
        
        WARNING: This method uses cls(**config) which assumes the saved config
        exactly matches the constructor signature. Subclasses that add required
        arguments should override this method. Consider using from_config() or
        storing class_name in checkpoints for more robust loading.
        
        Args:
            path: Path to the saved agent
            device: Device to load the agent to
            
        Returns:
            Loaded agent instance
            
        Raises:
            TypeError: If config doesn't match constructor (subclass mismatch)
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        try:
            agent = cls(**config)
        except TypeError as e:
            raise TypeError(
                f"Failed to load agent: config doesn't match {cls.__name__} constructor. "
                f"Subclasses should override load() if they add required arguments. "
                f"Original error: {e}"
            ) from e
        agent.load_state_dict(checkpoint["state_dict"])
        return agent
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for this agent.
        
        Returns:
            Configuration dictionary
        """
        pass
    
    def reset_hidden(self):
        """
        Reset any hidden state (for recurrent agents).
        
        IMPORTANT: Recurrent agents (e.g., RL2Agent) MUST override this method
        to properly reset their hidden state, previous action, reward, and any
        other episode-dependent state. Failure to properly reset will cause
        information leakage between tasks/episodes.
        
        The default implementation is a no-op for non-recurrent agents.
        """
        pass


class RandomAgent(BaseAgent):
    """
    Random agent baseline that selects actions uniformly at random.
    """
    
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def encode_observation(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        return torch.zeros(1)
    
    def select_action(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_actions = len(admissible_commands)
        if num_actions == 0:
            return 0, torch.tensor(0.0), torch.tensor(0.0)
        
        action_idx = torch.randint(0, num_actions, (1,)).item()
        log_prob = torch.tensor(-torch.log(torch.tensor(float(num_actions))))
        value = torch.tensor(0.0)
        
        return action_idx, log_prob, value
    
    def evaluate_actions(
        self,
        observations: List[Dict[str, str]],
        admissible_commands_list: List[List[str]],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(observations)
        
        log_probs = []
        for i, cmds in enumerate(admissible_commands_list):
            num_actions = len(cmds)
            log_probs.append(-torch.log(torch.tensor(float(max(1, num_actions)))))
        
        log_probs = torch.stack(log_probs)
        values = torch.zeros(batch_size)
        entropies = torch.tensor([torch.log(torch.tensor(float(max(1, len(cmds))))) 
                                  for cmds in admissible_commands_list])
        
        return log_probs, values, entropies
    
    def get_value(
        self,
        observation: Dict[str, str],
        admissible_commands: List[str]
    ) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def get_config(self) -> Dict[str, Any]:
        return {}
