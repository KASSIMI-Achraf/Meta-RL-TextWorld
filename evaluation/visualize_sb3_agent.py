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

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.textworld_env import TextWorldEnv
from utils.sb3_wrappers import TextWorldEncodingWrapper
from agents.sb3_policy import TextWorldDistilBertPolicy


# Must match training config exactly
MILD_REWARD_SHAPING = {
    "win_bonus": 50.0,
    "score_multiplier": 10.0,
    "exploration_bonus": 0.2,
    "inventory_bonus": 0.5,
    "time_penalty": -0.01,
    "productive_action": 0.05,
    "revisit_penalty_scale": 0.1,
    "loss_penalty": -1.0,
    "action_repeat_penalty": -0.05,
}


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

    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        base_env = TextWorldEnv(
            game_path=str(game_path),
            max_steps=75,  # Match training config
            use_admissible_commands=True,
            reward_shaping=MILD_REWARD_SHAPING
        )
        return TextWorldEncodingWrapper(base_env, device=device)
    
    vec_env = DummyVecEnv([make_env])
    
    # Get underlying env for accessing game info
    base_env = vec_env.envs[0].env  # TextWorldEnv underneath the wrapper
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path, device=device)
    
    # Debug: print observation space info
    print(f"\nDEBUG - Model observation space: {model.observation_space}")
    print(f"DEBUG - Env observation space: {vec_env.observation_space}")
    
    # Run episodes
    for ep in range(args.episodes):
        print(f"\n\n{'='*60}")
        print(f"EPISODE {ep+1}")
        print(f"{'='*60}")
        
        obs = vec_env.reset()  
        done = False
        steps = 0
        total_reward = 0
        
        # Debug: Print observation keys on first step
        print(f"\nDEBUG - Observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")
        
        # Print initial state
        print(f"\n{base_env._current_infos.get('description', '')}")
        print(f"\nAdmissible commands: {base_env.get_admissible_commands()}")
        
        while not done:
            # Add delay for readability
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Predict action - also get action probabilities for debugging
            action, _states = model.predict(obs, deterministic=True)
            
            # Debug: Get action distribution
            with torch.no_grad():
                obs_tensor = {k: torch.as_tensor(v).to(device) for k, v in obs.items()}
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs[0].cpu().numpy()
                top_actions = np.argsort(probs)[-5:][::-1]  # Top 5 actions
            
            # Get command name before stepping
            admissible = base_env.get_admissible_commands()
            action_idx = int(action[0])  # VecEnv returns array
            command = admissible[action_idx] if action_idx < len(admissible) else "???"
            
            # Debug: show action probabilities (first few steps only)
            if steps < 5:
                print(f"DEBUG - Action probs (top 5): {[(admissible[i] if i < len(admissible) else '?', f'{probs[i]:.3f}') for i in top_actions]}")
            
            # Step environment - VecEnv returns (obs, rewards, dones, infos)
            obs, rewards, dones, infos = vec_env.step(action)
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            
            total_reward += reward
            steps += 1
            
            # Print feedback
            print("-" * 40)
            print(f"Step {steps}: > {command}")
            print(f"Reward: {reward:.2f} (Total: {total_reward:.2f})")
            print("-" * 40)
            # Get feedback from underlying env
            feedback = base_env._current_obs if base_env._current_obs else ""
            if feedback:
                print(feedback[:500])  # Truncate long feedback
            
            if done:
                won = info.get('won', False)
                print(f"\n{'*'*40}")
                print(f"GAME OVER: {'WON!' if won else 'LOST'}")
                print(f"Total Steps: {steps}")
                print(f"Final Reward: {total_reward:.2f}")
                print(f"{'*'*40}")
    
    vec_env.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize SB3 Agent")
    parser.add_argument("--game", type=str, required=True, help="Path to game file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps in seconds")
    
    args = parser.parse_args()
    visualize(args)

if __name__ == "__main__":
    main()
