"""
Simple SB3 PPO Training Script for Single Game

This script trains a Stable Baselines3 PPO agent on a single TextWorld game
with milder reward penalties designed to encourage exploration without
being overly punitive.

Usage:
    python training/train_sb3_single.py --game games/train/train_0000.z8
    python training/train_sb3_single.py --game games/train/train_0000.z8 --episodes 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from envs.textworld_env import TextWorldEnv
from utils.sb3_wrappers import TextWorldEncodingWrapper
from agents.sb3_policy import TextWorldDistilBertPolicy


# Mild reward shaping configuration
MILD_REWARD_SHAPING = {
    "win_bonus": 50.0,
    "score_multiplier": 10.0,
    "exploration_bonus": 0.2,
    "inventory_bonus": 0.5,
    "time_penalty": -0.01,
    "productive_action": 0.05,
    "revisit_penalty_scale": 0.003, # Drastically reduced (was 0.1) to prevent -200 loops
    "loss_penalty": -1.0,
    "action_repeat_penalty": -0.05,
}


class EpisodeLoggerCallback(BaseCallback):
    """Callback to log episode results and stop after consecutive wins."""
    
    def __init__(self, consecutive_wins_target: int = 10, verbose=1):
        super().__init__(verbose)
        self.consecutive_wins_target = consecutive_wins_target
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_wins = []
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
    
    def _on_step(self) -> bool:
        # Check if episode ended
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                
                # Get info from the environment
                infos = self.locals.get("infos", [])
                info = infos[i] if i < len(infos) else {}
                
                # Get episode reward from monitor wrapper
                ep_reward = info.get("episode", {}).get("r", 0)
                won = info.get("won", False)
                
                self.episode_rewards.append(ep_reward)
                self.episode_wins.append(won)
                
                # Track consecutive wins
                if won:
                    self.consecutive_wins += 1
                    self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
                else:
                    self.consecutive_wins = 0
                
                # Print episode result
                status = "WON" if won else "LOST"
                print(f"Episode {self.episode_count}: {status} | Reward: {ep_reward:.2f} | Consecutive Wins: {self.consecutive_wins}")
                
                # Stop if we've reached target consecutive wins
                if self.consecutive_wins >= self.consecutive_wins_target:
                    print(f"\n*** TARGET REACHED: {self.consecutive_wins} consecutive wins! ***")
                    return False
        
        return True
    
    def _on_training_end(self):
        # Print summary
        total_wins = sum(self.episode_wins)
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {self.episode_count}")
        print(f"Total Wins: {total_wins} ({100*total_wins/max(1,self.episode_count):.1f}%)")
        print(f"Max Consecutive Wins: {self.max_consecutive_wins}")
        print(f"Average Reward: {avg_reward:.2f}")


def make_env(game_path: str, device: str):
    """Create a training environment with mild reward shaping."""
    def _init():
        from stable_baselines3.common.monitor import Monitor
        
        env = TextWorldEnv(
            game_path=game_path,
            max_steps=75,
            use_admissible_commands=True,
            reward_shaping=MILD_REWARD_SHAPING
        )
        
        env = TextWorldEncodingWrapper(env, device=device)
        
        log_dir = PROJECT_ROOT / "logs" / "sb3_single"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(log_dir / "monitor"))
        
        return env
    return _init


def train(args):
    """Run training."""
    print("=" * 60)
    print("SB3 PPO SINGLE GAME TRAINING")
    print("=" * 60)
    
    # Check if game exists
    game_path = Path(args.game)
    if not game_path.exists():
        game_path = PROJECT_ROOT / args.game
        if not game_path.exists():
            print(f"Error: Game file not found: {args.game}")
            sys.exit(1)
    
    game_path = str(game_path.absolute())
    print(f"Game: {game_path}")
    print(f"Target: {args.consecutive_wins} consecutive wins")
    
    # Device setup
    num_gpus = torch.cuda.device_count()
    device = "cuda:0" if num_gpus > 0 else "cpu"
    print(f"Device: {device}")
    
    # Print reward shaping config
    print("\nReward Shaping:")
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
        verbose=0,  # Reduce SB3 verbosity
        learning_rate=args.lr,
        n_steps=128,
        batch_size=64,
        device=device,
        tensorboard_log=str(PROJECT_ROOT / "logs" / "sb3_single_tensorboard")
    )
    
    # Create callback
    callback = EpisodeLoggerCallback(consecutive_wins_target=args.consecutive_wins)
    
    # Training - run until consecutive wins target is reached
    print(f"\nStarting training (target: {args.consecutive_wins} consecutive wins)...\n")
    model.learn(
        total_timesteps=10_000_000,  # Large limit, callback will stop early
        callback=callback,
        progress_bar=False
    )
    
    # Save model
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    game_name = Path(game_path).stem
    save_path = save_dir / f"sb3_ppo_{game_name}"
    model.save(str(save_path))
    print(f"\nModel saved to: {save_path}")
    
    # Cleanup
    env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train SB3 PPO on a single TextWorld game"
    )
    
    parser.add_argument(
        "--game", 
        type=str, 
        required=True,
        help="Path to the TextWorld game file (.z8)"
    )
    parser.add_argument(
        "--consecutive_wins", 
        type=int, 
        default=10,
        help="Stop after this many consecutive wins (default: 10)"
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
