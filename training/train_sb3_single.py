"""
Simple SB3 PPO Training Script for Single Game

This script trains a Stable Baselines3 PPO agent on a single TextWorld game
with milder reward penalties designed to encourage exploration without
being overly punitive.

Usage:
    python training/train_sb3_single.py --game games/train/train_0000.z8
    python training/train_sb3_single.py --game games/train/train_0000.z8 --timesteps 50000
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.textworld_env import TextWorldEnv
from utils.sb3_wrappers import TextWorldEncodingWrapper, TextWorldTrialEnv
from agents.sb3_policy import TextWorldDistilBertPolicy


# Mild reward shaping configuration
MILD_REWARD_SHAPING = {
    "win_bonus": 50.0,           # Keep win bonus high
    "score_multiplier": 10.0,    # Keep score reward high
    "exploration_bonus": 0.2,    # Slightly higher exploration bonus
    "inventory_bonus": 0.5,      # Keep inventory bonus
    "time_penalty": -0.01,       # Much milder time penalty (was -0.1)
    "productive_action": 0.05,   # Keep productive action bonus
    "revisit_penalty_scale": 0.1, # Much milder revisit penalty (was 0.5)
    "loss_penalty": -1.0,        # Much milder loss penalty (was -5.0)
    "action_repeat_penalty": -0.05,  # Much milder repeat penalty (was -0.3)
}


def make_env(game_path: str, device: str, rank: int = 0):
    """Create a training environment with mild reward shaping."""
    def _init():
        env = TextWorldEnv(
            game_path=game_path,
            max_steps=100,
            use_admissible_commands=True,
            reward_shaping=MILD_REWARD_SHAPING
        )
        
        env = TextWorldEncodingWrapper(env, device=device)
        env = TextWorldTrialEnv(env, episodes_per_trial=10)
        
        log_dir = PROJECT_ROOT / "logs" / "sb3_single"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(log_dir / f"monitor_{rank}"))
        
        return env
    return _init


def train(args):
    """Run training."""
    print("=" * 60)
    print("SB3 PPO SINGLE GAME TRAINING (MILD PENALTIES)")
    print("=" * 60)
    
    # Check if game exists
    game_path = Path(args.game)
    if not game_path.exists():
        # Try relative to project root
        game_path = PROJECT_ROOT / args.game
        if not game_path.exists():
            print(f"Error: Game file not found: {args.game}")
            sys.exit(1)
    
    game_path = str(game_path.absolute())
    print(f"Game: {game_path}")
    
    # Device setup
    num_gpus = torch.cuda.device_count()
    device = "cuda:0" if num_gpus > 0 else "cpu"
    print(f"Device: {device}")
    
    # Print reward shaping config
    print("\nReward Shaping (Mild Penalties):")
    for key, value in MILD_REWARD_SHAPING.items():
        print(f"  {key}: {value}")
    
    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([make_env(game_path, device)])
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        TextWorldDistilBertPolicy,
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=128,
        batch_size=64,
        device=device,
        tensorboard_log=str(PROJECT_ROOT / "logs" / "sb3_single_tensorboard")
    )
    
    # Training
    print(f"\nStarting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    
    # Save model
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    game_name = Path(game_path).stem
    save_path = save_dir / f"sb3_ppo_{game_name}_mild"
    model.save(str(save_path))
    print(f"\nModel saved to: {save_path}")
    
    # Cleanup
    env.close()
    
    print("\nTraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train SB3 PPO on a single TextWorld game with mild penalties"
    )
    
    parser.add_argument(
        "--game", 
        type=str, 
        required=True,
        help="Path to the TextWorld game file (.z8)"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=10000,
        help="Total training timesteps (default: 10000)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
