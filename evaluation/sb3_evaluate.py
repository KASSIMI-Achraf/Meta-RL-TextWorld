"""
SB3 Agent Evaluation

Evaluate a trained Stable Baselines3 model on TextWorld games.
Reports success rate, mean reward, and episode statistics.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

print("Starting evaluation script...", flush=True)

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Loading SB3...", flush=True)
from stable_baselines3 import PPO

print("Loading TextWorld env...", flush=True)
from envs.textworld_env import TextWorldEnv

print("Loading SB3 wrappers...", flush=True)
from utils.sb3_wrappers import TextWorldEncodingWrapper, TextWorldTrialEnv

print("Loading DistilBERT encoder...", flush=True)
from agents.text_encoder import DistilBERTEncoder

print("All modules loaded!", flush=True)

def evaluate_sb3_agent(
    model_path: str,
    game_paths: List[str],
    num_episodes_per_game: int = 5,
    max_steps: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained SB3 model on given games.
    
    Args:
        model_path: Path to the saved SB3 model (.zip file)
        game_paths: List of paths to game files
        num_episodes_per_game: Number of episodes to run per game
        max_steps: Maximum steps per episode
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...", flush=True)
    model = PPO.load(model_path)
    if verbose:
        print("Model loaded successfully!", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}", flush=True)
    
    # Results tracking
    all_rewards = []
    all_steps = []
    all_wins = []
    all_losses = []
    game_results = {}
    
    total_games = len(game_paths)
    
    # Pre-load encoder once (this is the slow part)
    if verbose:
        print("Initializing text encoder (this may take a moment)...", flush=True)
    
    encoder = DistilBERTEncoder(
        model_name="distilbert-base-uncased",
        freeze_layers=4,
        output_size=768,
        device=device
    )
    encoder.eval()
    
    for game_idx, game_path in enumerate(game_paths):
        game_name = Path(game_path).stem
        
        if verbose:
            print(f"\n[{game_idx + 1}/{total_games}] Evaluating on: {game_name}", flush=True)
        
        game_rewards = []
        game_steps = []
        game_wins = []
        game_losses = []
        
        # Create environment 
        # Need to use TrialEnv wrapper to match training observation space
        # (adds prev_action, prev_reward, trial_time to observations)
        env = TextWorldEnv(
            game_path=game_path,
            max_steps=max_steps,
            use_admissible_commands=True
        )
        # Pass shared encoder instance
        env = TextWorldEncodingWrapper(env, device=device, encoder=encoder)
        # Use episodes_per_trial=1 to get proper per-episode metrics
        env = TextWorldTrialEnv(env, episodes_per_trial=1)
        
        for ep in range(num_episodes_per_game):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            won = False
            lost = False
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated
                
                if info.get("won", False):
                    won = True
                if info.get("lost", False):
                    lost = True
            
            game_rewards.append(episode_reward)
            game_steps.append(episode_steps)
            game_wins.append(won)
            game_losses.append(lost)
            
            if verbose:
                status = "WON" if won else ("LOST" if lost else "TIMEOUT")
                print(f"  Episode {ep + 1}: {status} | Steps: {episode_steps} | Reward: {episode_reward:.2f}", flush=True)
        
        env.close()
        
        # Store game results
        game_results[game_name] = {
            "mean_reward": np.mean(game_rewards),
            "mean_steps": np.mean(game_steps),
            "win_rate": np.mean(game_wins),
            "episodes": num_episodes_per_game
        }
        
        # Aggregate
        all_rewards.extend(game_rewards)
        all_steps.extend(game_steps)
        all_wins.extend(game_wins)
        all_losses.extend(game_losses)
    
    # Calculate overall metrics
    results = {
        "overall": {
            "total_episodes": len(all_rewards),
            "total_games": total_games,
            "win_rate": np.mean(all_wins) * 100,
            "loss_rate": np.mean(all_losses) * 100,
            "timeout_rate": (1 - np.mean(all_wins) - np.mean(all_losses)) * 100,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_steps": np.mean(all_steps),
            "std_steps": np.std(all_steps),
        },
        "per_game": game_results
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    overall = results["overall"]
    
    print(f"\nTotal Episodes: {overall['total_episodes']}")
    print(f"Total Games: {overall['total_games']}")
    print("-" * 40)
    print(f"Win Rate:     {overall['win_rate']:.1f}%")
    print(f"Loss Rate:    {overall['loss_rate']:.1f}%")
    print(f"Timeout Rate: {overall['timeout_rate']:.1f}%")
    print("-" * 40)
    print(f"Mean Reward:  {overall['mean_reward']:.3f} ± {overall['std_reward']:.3f}")
    print(f"Mean Steps:   {overall['mean_steps']:.1f} ± {overall['std_steps']:.1f}")
    
    print("\n" + "-" * 40)
    print("Per-Game Breakdown:")
    print("-" * 40)
    
    for game_name, stats in results["per_game"].items():
        print(f"  {game_name}: Win {stats['win_rate']*100:.0f}% | "
              f"Reward {stats['mean_reward']:.2f} | "
              f"Steps {stats['mean_steps']:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SB3 agent on TextWorld games")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SB3 model (.zip)")
    parser.add_argument("--games", type=str, required=True,
                        help="Path to game file or directory containing games")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per game (default: 5)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode (default: 100)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Collect game files
    games_path = Path(args.games)
    if games_path.is_file():
        game_files = [str(games_path)]
    else:
        game_files = sorted(glob.glob(str(games_path / "*.z8")))
        game_files.extend(sorted(glob.glob(str(games_path / "*.ulx"))))
    
    if not game_files:
        print(f"Error: No game files found in {args.games}")
        sys.exit(1)
    
    print(f"Found {len(game_files)} game(s)")
    
    # Run evaluation
    results = evaluate_sb3_agent(
        model_path=args.model,
        game_paths=game_files,
        num_episodes_per_game=args.episodes,
        max_steps=args.max_steps,
        verbose=not args.quiet
    )
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
