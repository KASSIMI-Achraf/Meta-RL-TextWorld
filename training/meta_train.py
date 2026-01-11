"""
Meta-Training Script

Entry point for running meta-training with RL².
Handles configuration loading, environment setup, and training loop.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.textworld_env import TextWorldEnv
from envs.game_generator import generate_games
from agents.meta_rl_agent import RL2Agent
from meta_learning.rl2 import RL2
from utils.logger import Logger
from utils.helpers import set_seed, get_device, load_config


class MetaTrainer:
    """
    Orchestrates meta-training for text-based RL agents.
    
    Handles:
    - Configuration loading
    - Environment/game setup
    - RL² algorithm instantiation
    - Training loop with logging and checkpointing
    """
    
    def __init__(
        self,
        config_path: str,
        seed: int = 42,
        debug: bool = False
    ):
        """
        Initialize the meta-trainer.
        
        Args:
            config_path: Path to configuration file
            seed: Random seed
            debug: Enable debug mode with reduced iterations
        """
        self.config = load_config(config_path)
        self.seed = seed
        self.debug = debug
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Get device
        self.device = get_device(self.config.get("device", "auto"))
        
        # Initialize logger
        log_config = self.config.get("logging", {})
        self.logger = Logger(
            log_dir=log_config.get("log_dir", "logs"),
            use_tensorboard=log_config.get("tensorboard", True)
        )
        
        # Game/environment paths
        self.train_games = []
        self.val_games = []
        
        # Algorithm instance
        self.meta_learner = None
    
    def setup_environments(self, games_dir: str = "games"):
        """
        Load or generate games and create environments.
        
        Args:
            games_dir: Directory containing generated games
        """
        games_path = Path(games_dir)
        
        # Check if games exist, otherwise generate them
        if not (games_path / "train").exists():
            print("Games not found. Generating games...")
            self._generate_games(games_dir)
        
        # Load game paths
        train_dir = games_path / "train"
        val_dir = games_path / "val"
        
        self.train_games = list(train_dir.glob("*.z8")) + list(train_dir.glob("*.ulx"))
        self.val_games = list(val_dir.glob("*.z8")) + list(val_dir.glob("*.ulx"))
        
        print(f"Loaded {len(self.train_games)} training games")
        print(f"Loaded {len(self.val_games)} validation games")
        
        if len(self.train_games) == 0:
            raise RuntimeError("No training games found!")
    
    def _generate_games(self, output_dir: str):
        """Generate games if they don't exist."""
        from pathlib import Path
        env_config = self.config.get("game_generation", {})
        difficulty = self.config.get("default_difficulty", "easy")
        
        num_train = env_config.get("num_train_games", 100)
        num_val = env_config.get("num_val_games", 20)
        num_test = env_config.get("num_test_games", 30)
        
        base_dir = Path(output_dir)
        
        generate_games(num_train, base_dir / "train", difficulty, 
                      seed_offset=self.seed, prefix="train_")
        generate_games(num_val, base_dir / "val", difficulty, 
                      seed_offset=self.seed + num_train, prefix="val_")
        generate_games(num_test, base_dir / "test", difficulty, 
                      seed_offset=self.seed + num_train + num_val, prefix="test_")
    
    def _create_envs(self, game_paths: List[Path]) -> List[TextWorldEnv]:
        """Create TextWorld environments from game paths."""
        env_config = self.config.get("environment", {})
        max_steps = env_config.get("max_steps", 100)
        
        envs = []
        for game_path in game_paths:
            env = TextWorldEnv(
                game_path=str(game_path),
                max_steps=max_steps,
                use_admissible_commands=True
            )
            envs.append(env)
        
        return envs
    
    def setup_agent(self) -> None:
        """Create and configure the RL² agent."""
        agent_config = self.config.get("agent", {})
        
        # Encoder config
        encoder_type = agent_config.get("encoder_type", "tinybert")
        encoder_subconfig = agent_config.get(encoder_type, agent_config.get("tinybert", {}))
        encoder_config = {
            "model_name": encoder_subconfig.get("model_name", "huawei-noah/TinyBERT_General_4L_312D"),
            "freeze_layers": encoder_subconfig.get("freeze_layers", 2),
            "max_length": 512,
            "hidden_size": encoder_subconfig.get("hidden_size", 312),
        }
        
        # Policy/value network config
        policy_config = agent_config.get("policy", {})
        value_config = agent_config.get("value", {})
        rl2_config = self.config.get("rl2", {})
        
        self.agent = RL2Agent(
            encoder_config=encoder_config,
            policy_hidden_sizes=policy_config.get("hidden_sizes", [256, 128]),
            value_hidden_sizes=value_config.get("hidden_sizes", [256, 128]),
            hidden_size=encoder_config.get("hidden_size", 768),
            rnn_hidden_size=rl2_config.get("hidden_size", 256),
            device=str(self.device)
        )
        
        self.agent.to(self.device)
        print(f"Created RL2 agent on {self.device}")
    
    def setup_algorithm(self) -> None:
        """Initialize the RL² meta-learning algorithm."""
        meta_config = self.config.get("meta_learning", {})
        pg_config = self.config.get("policy_gradient", {})
        rl2_config = self.config.get("rl2", {})
        
        self.meta_learner = RL2(
            agent=self.agent,
            learning_rate=meta_config.get("outer_lr", 0.001),
            episodes_per_trial=rl2_config.get("episodes_per_trial", 10),
            meta_batch_size=meta_config.get("meta_batch_size", 4),
            gamma=pg_config.get("gamma", 0.99),
            gae_lambda=pg_config.get("gae_lambda", 0.95),
            entropy_coef=pg_config.get("entropy_coef", 0.01),
            value_coef=pg_config.get("value_coef", 0.5),
            max_grad_norm=1.0,
            device=str(self.device)
        )
        
        print("Initialized RL2 algorithm")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run the meta-training loop.
        
        Returns:
            Training history
        """
        train_envs = self._create_envs(self.train_games)
        val_envs = self._create_envs(self.val_games)
        
        train_config = self.config.get("training", {})
        checkpoint_config = self.config.get("checkpoints", {})
        meta_config = self.config.get("meta_learning", {})
        
        num_iterations = meta_config.get("num_iterations", 1000)
        if self.debug:
            num_iterations = min(num_iterations, 5)
        
        history = self.meta_learner.meta_train(
            train_envs=train_envs,
            val_envs=val_envs,
            num_iterations=num_iterations,
            val_every=train_config.get("val_every", 50),
            save_every=train_config.get("save_every", 100),
            save_dir=checkpoint_config.get("checkpoint_dir", "checkpoints"),
            logger=self.logger
        )
        
        for env in train_envs + val_envs:
            env.close()
        
        return history
    
    def run(self) -> Dict[str, List[float]]:
        """
        Full training pipeline.
        
        Returns:
            Training history
        """
        print("=" * 60)
        print("Meta-Training with RL2")
        print("=" * 60)
        
        self.setup_environments()
        self.setup_agent()
        self.setup_algorithm()
        
        history = self.train()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        if "val_reward" in history and len(history["val_reward"]) > 0:
            print(f"Best validation reward: {max(history['val_reward']):.3f}")
        
        return history


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-train an RL agent for TextWorld")
    parser.add_argument("--config", type=str, default="configs/meta_train.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    trainer = MetaTrainer(
        config_path=args.config,
        seed=args.seed,
        debug=args.debug
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
