"""
TextWorld Environment Wrapper

A Gymnasium-compatible wrapper for TextWorld games that provides:
- Standardized observation format (description, inventory, admissible commands)
- Reward normalization and shaping options
- Easy integration with meta-learning pipelines
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import os
from collections import defaultdict


class TextWorldEnv(gym.Env):
    """
    Gymnasium wrapper for TextWorld environments.
    
    This wrapper provides a consistent interface for interacting with TextWorld games,
    handling observation parsing, action space management, and episode termination.
    
    Attributes:
        game_path: Path to the TextWorld game file (.z8 or .ulx)
        max_steps: Maximum steps per episode
        use_admissible_commands: Whether to use admissible commands action space
        request_infos: Information to request from the game
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        game_path: str,
        max_steps: int = 100,
        use_admissible_commands: bool = True,
        max_admissible_commands: int = 20,
        request_infos: Optional[Any] = None,  
        render_mode: Optional[str] = None,
        reward_shaping: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the TextWorld environment.
        
        Args:
            game_path: Path to the TextWorld game file
            max_steps: Maximum number of steps per episode
            use_admissible_commands: If True, action space is discrete over admissible commands
            max_admissible_commands: Maximum number of admissible commands to consider
            request_infos: TextWorld info request configuration
            render_mode: Gymnasium render mode
            reward_shaping: Dictionary of reward shaping coefficients:
                - win_bonus: Bonus for winning (default: 10.0)
                - score_multiplier: Multiplier for score gain (default: 10.0)
                - exploration_bonus: Reward for finding new location (default: 0.1)
                - inventory_bonus: Reward for finding new item (default: 0.5)
                - time_penalty: Penalty per step (default: -0.01)
                - productive_action: Reward for state-changing action (default: 0.05)
        """
        super().__init__()
        
        self.game_path = game_path
        self.max_steps = max_steps
        self.use_admissible_commands = use_admissible_commands
        self.max_admissible_commands = max_admissible_commands
        self.render_mode = render_mode
        
        # Reward shaping config
        self.shaping_config = {
            "win_bonus": 50.0,
            "score_multiplier": 10.0,
            "exploration_bonus": 0.1,
            "inventory_bonus": 0.5,
            "time_penalty": -0.1,
            "productive_action": 0.05,
            "revisit_penalty_scale": 0.5,  # Penalty = (visits - 1) * scale (increased for loop prevention)
            "loss_penalty": -5.0,  # Penalty for losing the game
            "action_repeat_penalty": -0.3,  # Penalty for repeating same action at same location
        }
        if reward_shaping:
            self.shaping_config.update(reward_shaping)
            
        # State tracking for shaping
        self._visited_locations = set()
        self._location_visit_counts = defaultdict(int)
        self._seen_inventory_items = set()
        self._last_score = 0
        self._last_action_at_location = {}  # Track last action per location for loop detection
        
        self._request_infos_arg = request_infos
        self.request_infos = None
        self._env = None
        self._current_step = 0
        self._current_obs = None
        self._current_infos = None
        self._admissible_commands = []
        
        # Define observation and action spaces
        # Observation is a dictionary with text fields
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=4096),
            "description": spaces.Text(max_length=2048),
            "inventory": spaces.Text(max_length=1024),
            "feedback": spaces.Text(max_length=1024),
        })
        
        # Action space depends on mode
        if use_admissible_commands:
            # Discrete action space over admissible commands
            self.action_space = spaces.Discrete(max_admissible_commands)
        else:
            # Text action space for free-form commands
            self.action_space = spaces.Text(max_length=256)
    
    def _create_env(self):
        """Create the underlying TextWorld environment."""
        if self._env is not None:
            self._env.close()
        
        import textworld
        import textworld.gym

        if self.request_infos is None:
            if self._request_infos_arg is None:
                self.request_infos = textworld.EnvInfos(
                    description=True,
                    inventory=True,
                    admissible_commands=True,
                    entities=True,
                    verbs=True,
                    extras=["walkthrough"],
                    score=True,
                    max_score=True,
                    won=True,
                    lost=True
                )
            else:
                self.request_infos = self._request_infos_arg
                # Ensure critical info is requested for shaping
                self.request_infos.score = True
                self.request_infos.max_score = True
                self.request_infos.won = True
                self.request_infos.lost = True
                self.request_infos.description = True
                self.request_infos.inventory = True
        
        # Register and create the TextWorld gym environment
        # TextWorld uses its own gym wrapper, not gymnasium
        env_id = textworld.gym.register_game(
            self.game_path,
            request_infos=self.request_infos,
            max_episode_steps=self.max_steps
        )
        # Use textworld.gym.make instead of gymnasium.make
        self._env = textworld.gym.make(env_id)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Dictionary containing text observations
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Create environment if needed
        if self._env is None:
            self._create_env()
        
        obs, infos = self._env.reset()
        
        self._current_step = 0
        self._current_obs = obs
        self._current_infos = infos
        self._admissible_commands = infos.get("admissible_commands", [])
        
        # Reset shaping state
        self._visited_locations = set()
        self._location_visit_counts = defaultdict(int)
        self._seen_inventory_items = set()
        self._last_score = infos.get("score", 0)
        self._last_action_at_location = {}  # Reset action tracking
        
        # Record initial state
        if "description" in infos:
            self._visited_locations.add(infos["description"])
            self._location_visit_counts[infos["description"]] = 1
        if "inventory" in infos:
            self._seen_inventory_items.add(infos["inventory"])
        
        # Build structured observation
        observation = self._build_observation(obs, infos)
        
        # Build info dictionary
        info = {
            "admissible_commands": self._admissible_commands,
            "entities": infos.get("entities", []),
            "verbs": infos.get("verbs", []),
            "walkthrough": infos.get("extra.walkthrough", []),
            "won": False,
            "lost": False,
        }
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict[str, str], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Either an integer index (if use_admissible_commands) or a string command
            
        Returns:
            observation: Dictionary containing text observations
            reward: Reward for this step
            terminated: Whether the episode ended (win/loss)
            truncated: Whether the episode was truncated (max steps)
            info: Additional information dictionary
        """
        if self.use_admissible_commands:
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action[0])
            elif isinstance(action, (int, np.integer)):
                action = int(action)
            else:
                command = str(action)
                action = None
            
            if action is not None:
                if action < len(self._admissible_commands):
                    command = self._admissible_commands[action]
                else:
                    command = self._admissible_commands[0] if self._admissible_commands else "look"
        else:
            command = str(action) if not isinstance(action, str) else action
        
        obs, raw_reward, done, infos = self._env.step(command)
        
        self._current_step += 1
        self._current_obs = obs
        self._current_infos = infos
        self._admissible_commands = infos.get("admissible_commands", [])
        
        won = infos.get("won", False)
        lost = infos.get("lost", False)
        terminated = done and (won or lost)
        truncated = done and not terminated
        
        shaped_reward = 0.0
        
        shaped_reward += self.shaping_config["time_penalty"]
        
        current_score = infos.get("score", 0)
        score_diff = current_score - self._last_score
        if score_diff > 0:
            shaped_reward += score_diff * self.shaping_config["score_multiplier"]
        self._last_score = current_score
        
        if won:
            shaped_reward += self.shaping_config["win_bonus"]
        
        # Loss penalty for losing the game
        if lost:
            shaped_reward += self.shaping_config.get("loss_penalty", -5.0)
        
        description = infos.get("description", "")
        if description:
            # Revisit penalty: scaled by number of times visited minus 1
            # 1st visit: 0 penalty
            # 2nd visit: 1 * scale
            # 3rd visit: 2 * scale, etc.
            self._location_visit_counts[description] += 1
            visits = self._location_visit_counts[description]
            if visits > 1:
                revisit_penalty = (visits - 1) * self.shaping_config.get("revisit_penalty_scale", 0.5)
                # Apply penalty (subtract from reward)
                shaped_reward -= revisit_penalty
            
            # Action repeat penalty: penalize doing same action at same location
            loc_key = description
            if loc_key in self._last_action_at_location:
                if self._last_action_at_location[loc_key] == command:
                    shaped_reward += self.shaping_config.get("action_repeat_penalty", -0.3)
            self._last_action_at_location[loc_key] = command
        
        if description and description not in self._visited_locations:
            shaped_reward += self.shaping_config["exploration_bonus"]
            self._visited_locations.add(description)
            
        inventory = infos.get("inventory", "")
        if inventory and inventory not in self._seen_inventory_items:
            shaped_reward += self.shaping_config["inventory_bonus"]
            self._seen_inventory_items.add(inventory)
            
        final_reward = shaped_reward
        
        observation = self._build_observation(obs, infos)
        
        info = {
            "admissible_commands": self._admissible_commands,
            "entities": infos.get("entities", []),
            "verbs": infos.get("verbs", []),
            "command": command,
            "won": won,
            "lost": lost,
            "score": infos.get("score", 0),
            "max_score": infos.get("max_score", 1),
            "raw_reward": raw_reward, 
            "shaped_reward": shaped_reward
        }
        
        return observation, final_reward, terminated, truncated, info
    
    def _build_observation(self, obs: str, infos: Dict) -> Dict[str, str]:
        """
        Build a structured observation dictionary.
        
        Args:
            obs: Raw observation string from TextWorld
            infos: Info dictionary from TextWorld
            
        Returns:
            Structured observation dictionary
        """
        description = infos.get("description", "")
        inventory = infos.get("inventory", "")
        
        # Combine into full text observation
        full_text = f"DESCRIPTION: {description}\n\nINVENTORY: {inventory}\n\nFEEDBACK: {obs}"
        
        return {
            "text": full_text,
            "description": description,
            "inventory": inventory,
            "feedback": obs,
        }
    
    def get_admissible_commands(self) -> List[str]:
        """Get the current list of admissible commands."""
        return self._admissible_commands.copy()
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(self._current_obs)
        elif self.render_mode == "ansi":
            return self._current_obs
    
    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def get_game_info(self) -> Dict[str, Any]:
        """Get game metadata."""
        return {
            "game_path": self.game_path,
            "max_steps": self.max_steps,
            "use_admissible_commands": self.use_admissible_commands,
        }


class TextWorldBatchEnv:
    """
    Manages a batch of TextWorld environments for parallel task sampling.
    
    This is useful for meta-learning where we need to sample multiple tasks
    and collect trajectories in parallel.
    """
    
    def __init__(
        self,
        game_paths: List[str],
        max_steps: int = 100,
        use_admissible_commands: bool = True,
        max_admissible_commands: int = 20
    ):
        """
        Initialize the batch environment.
        
        Args:
            game_paths: List of paths to TextWorld game files
            max_steps: Maximum steps per episode
            use_admissible_commands: Whether to use admissible commands
            max_admissible_commands: Maximum admissible commands
        """
        self.game_paths = game_paths
        self.max_steps = max_steps
        self.use_admissible_commands = use_admissible_commands
        self.max_admissible_commands = max_admissible_commands
        
        self._envs = {}
    
    def get_env(self, game_path: str) -> TextWorldEnv:
        """
        Get or create an environment for a specific game.
        
        Args:
            game_path: Path to the game file
            
        Returns:
            TextWorldEnv instance
        """
        if game_path not in self._envs:
            self._envs[game_path] = TextWorldEnv(
                game_path=game_path,
                max_steps=self.max_steps,
                use_admissible_commands=self.use_admissible_commands,
                max_admissible_commands=self.max_admissible_commands
            )
        return self._envs[game_path]
    
    def sample_task(self) -> TextWorldEnv:
        """
        Sample a random task (game) from the available games.
        
        Returns:
            TextWorldEnv instance for the sampled game
        """
        game_path = np.random.choice(self.game_paths)
        return self.get_env(game_path)
    
    def close_all(self):
        """Close all environments."""
        for env in self._envs.values():
            env.close()
        self._envs = {}
