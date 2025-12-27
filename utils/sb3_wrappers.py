
import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import torch

from envs.textworld_env import TextWorldEnv
class TextWorldEncodingWrapper(gym.Wrapper):
    """
    Wrapper that encodes text observations into dense vectors using DistilBERT.
    
    This is necessary for SB3 compatibility, as SB3 replay buffers generally 
    expect numeric tensors/arrays, not strings.
    """
    
    def __init__(self, env: gym.Env, encoder_kwargs: Optional[Dict[str, Any]] = None, device: str = "cpu", encoder=None):
        super().__init__(env)
        self.device = device
        
        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            if encoder_kwargs is None:
                 encoder_kwargs = {
                     "model_name": "distilbert-base-uncased",
                     "freeze_layers": 4,
                     "output_size": 768
                 }
            from agents.text_encoder import DistilBERTEncoder
            self.encoder = DistilBERTEncoder(**encoder_kwargs, device=device)
            self.encoder.to(device)
            self.encoder.eval() # Inference mode
        
        # Update observation space
        # Original space has 'text', 'description', etc.
        # We replace/augment with 'obs_encoding'
        
        # Get hidden size
        hidden_size = self.encoder.hidden_size
        
        new_spaces = {**env.observation_space.spaces}
        
        # Add encoding space
        # Shape is (hidden_size,)
        new_spaces["obs_encoding"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(hidden_size,), dtype=np.float32
        )
        
        # Remove string fields to avoid compatibility issues with SB3 buffers
        keys_to_remove = ["text", "description", "inventory", "feedback"]
        for k in keys_to_remove:
            if k in new_spaces:
                del new_spaces[k]
                
        self.observation_space = gym.spaces.Dict(new_spaces)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(obs), reward, terminated, truncated, info
        
    def _encode(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode text fields."""
        # Clean current text fields from obs dict if we removed them from space
        # But we need their content to encode.
        
        full_text = obs.get("text", "")
        if not full_text:
             full_text = (
                f"Description: {obs.get('description', '')}\n"
                f"Inventory: {obs.get('inventory', '')}\n"
                f"Feedback: {obs.get('feedback', '')}"
             )
        
        # Encode
        with torch.no_grad():
            encoding = self.encoder.encode(full_text)
            # encoding is Tensor (hidden_size,) on device
            # Convert to numpy for Gym/SB3
            encoding_np = encoding.cpu().numpy()
            
        # Create new obs dict
        new_obs = {}
        # Copy other fields (like our meta-info 'prev_action' etc.)
        for k, v in obs.items():
            if k in self.observation_space.spaces:
                new_obs[k] = v
                
        new_obs["obs_encoding"] = encoding_np
        
        return new_obs


class TextWorldTrialEnv(gym.Wrapper):
    """
    Wrapper that treats a sequence of episodes (a trial) as a single long episode.
    
    This is useful for RL^2 meta-learning, where the agent's hidden state 
    needs to be maintained across standard episode boundaries required by the game.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        episodes_per_trial: int = 10
    ):
        super().__init__(env)
        self.episodes_per_trial = episodes_per_trial
        self.current_episode = 0
        
        # Track previous action/reward for meta-learning
        self.prev_action = 0
        self.prev_reward = 0.0
        
        # Define observation space to include meta-info
        # We need to flatten the observation because SB3 implementation of some buffers 
        # might struggle with complex nested dicts if not carefully handled.
        # But our custom Policy will handle the dict.
        # We add 'prev_action' and 'prev_reward' to the dict.
        
        # Copy original space
        self.observation_space = env.observation_space
        
        # Initialize observation space with meta-learning fields
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "prev_action": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "prev_reward": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "trial_time": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        """Reset the trial."""
        self.current_episode = 0
        self.prev_action = 0.0 # Assuming float for box
        self.prev_reward = 0.0
        
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), info

    def step(self, action):
        """Step the environment, handling episode boundaries internally."""
        # Map action if needed (assuming action is already compatible with inner env)
        # Store action for next step
        try:
            # If action is tensor/array
            self.prev_action = float(action)
        except:
            # If discrete (int)
            self.prev_action = float(action) if isinstance(action, (int, np.integer)) else 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.prev_reward = float(reward)
        
        done = terminated or truncated
        
        if done:
            self.current_episode += 1
            if self.current_episode < self.episodes_per_trial:
                # Soft reset: reset underlying game but continue "trial" episode
                # We need to maintain hidden state in agent (handled by policy usually)
                # But here we just provide the next observation from a fresh game state
                obs, reset_info = self.env.reset()
                info.update(reset_info) # Merge infos
                
    
                info["episode_done"] = True
                
                return self._wrap_obs(obs), reward, False, False, info
            else:
                # Trial is over
                return self._wrap_obs(obs), reward, True, False, info
                
        return self._wrap_obs(obs), reward, terminated, truncated, info

    def _wrap_obs(self, obs):
        """Add meta-info to observation."""
        obs["prev_action"] = np.array([self.prev_action], dtype=np.float32)
        obs["prev_reward"] = np.array([self.prev_reward], dtype=np.float32)
        obs["trial_time"] = np.array([self.current_episode / self.episodes_per_trial], dtype=np.float32)
        return obs

