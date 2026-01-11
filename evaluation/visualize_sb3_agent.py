"""
Visualize SB3 PPO Agent Gameplay

This script loads a trained SB3 PPO model and runs it on a TextWorld game,
printing the step-by-step interaction to the console.

Usage:
    python evaluation/visualize_sb3_agent.py --game games/train/train_0000.z8 --model checkpoints/sb3_ppo_train_0000
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO

from envs.textworld_env import TextWorldEnv
from utils.sb3_wrappers import TextWorldEncodingWrapper
from agents.sb3_policy import TextWorldDistilBertPolicy

def visualize(args):
    print("=" * 60)
    print("VISUALIZING AGENT GAMEPLAY")
    print("=" * 60)
    
    # Check paths
    game_path = Path(args.game)
    if not game_path.exists():
        game_path = PROJECT_ROOT / args.game
    
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
        # Try adding .zip extension if missing
        if not model_path.exists() and not str(model_path).endswith('.zip'):
            model_path = model_path.with_suffix('.zip')
            
    if not game_path.exists():
        print(f"Error: Game file not found: {args.game}")
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Game: {game_path}")
    print(f"Model: {model_path}")
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create environment
    # Note: We use the same wrappers as training to ensure observation compatibility
    env = TextWorldEnv(
        game_path=str(game_path),
        max_steps=72,  # Match training config
        use_admissible_commands=True,
        request_infos=None, # Use default infos
        render_mode="human"
    )
    env = TextWorldEncodingWrapper(env, device=device)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path, device=device)
    
    # Run episodes
    for ep in range(args.episodes):
        print(f"\n\n--- EPISODE {ep+1} ---")
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        # Print initial state
        env.unwrapped.render()
        
        while not done:
            # Add delay for readability
            time.sleep(1.0)
            
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Print feedback
            print("-" * 40)
            print(f"Step: {steps} | Reward: {reward:.2f}")
            print(f"Command: > {info.get('command', '???')}")
            print("-" * 40)
            print(info.get('feedback', ''))
            
            if done:
                print(f"\nGame Over! Result: {'WON' if info.get('won') else 'LOST'}")
                print(f"Total Steps: {steps}")
                print(f"Total Reward: {total_reward:.2f}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize SB3 Agent")
    parser.add_argument("--game", type=str, required=True, help="Path to game file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    
    args = parser.parse_args()
    visualize(args)

if __name__ == "__main__":
    main()
