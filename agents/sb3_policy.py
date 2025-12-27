
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from agents.text_encoder import DistilBERTEncoder

class TextWorldFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor for TextWorld.
    
    Uses DistilBERT to encode text observations and concatenates them with 
    meta-info (prev_action, prev_reward).
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        device: str = "cpu",
        hidden_size: int = 768,
        encoder_config: Optional[Dict[str, Any]] = None
    ):
        # Calculate features dim
        # DistilBERT size + meta-info sizes
        features_dim = hidden_size
        if "prev_action" in observation_space.spaces:
            features_dim += observation_space.spaces["prev_action"].shape[0] # assuming Box(1,)
        if "prev_reward" in observation_space.spaces:
            features_dim += observation_space.spaces["prev_reward"].shape[0]
        if "trial_time" in observation_space.spaces:
            features_dim += observation_space.spaces["trial_time"].shape[0]
            
        super().__init__(observation_space, features_dim)
        
        self.device = device
        
        # Default config
        if encoder_config is None:
            encoder_config = {
                "model_name": "distilbert-base-uncased",
                "freeze_layers": 4,
                "max_length": 512,
                "hidden_size": 768,
                "output_size": hidden_size
            }
            
        self.encoder = DistilBERTEncoder(**encoder_config, device=device)
        # Move encoder to device
        self.encoder.to(device)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Expects 'obs_encoding' tensor in observations dict (pre-encoded text).
        Concatenates text encoding with meta-info (prev_action, prev_reward).
        """
        
        # Assume 'obs_encoding' key exists and is a tensor from our wrapper
        if "obs_encoding" in observations:
             text_features = observations["obs_encoding"]
        else:
             # Fallback if we passed raw text (won't work with standard buffer usually)
             # but let's assume valid tensor setup
             raise ValueError("Observation must contain 'obs_encoding' tensor. Use TextWorldEncodingWrapper.")

        # Concatenate with meta-info
        features = [text_features]
        
        if "prev_action" in observations:
            features.append(observations["prev_action"])
        if "prev_reward" in observations:
            features.append(observations["prev_reward"])
        if "trial_time" in observations:
            features.append(observations["trial_time"])
            
        return torch.cat(features, dim=-1)


class TextWorldDistilBertPolicy(ActorCriticPolicy):
    """
    Custom Policy for TextWorld using DistilBERT features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # We enforce our custom extractor
        kwargs["features_extractor_class"] = TextWorldFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        
        if net_arch is None:
             net_arch = [dict(pi=[256, 128], vf=[256, 128])]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
