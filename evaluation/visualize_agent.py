"""
Agent Gameplay Visualization

Watch a trained SB3 agent play a TextWorld game step-by-step.
Displays game state, admissible commands, and agent's chosen actions.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from envs.textworld_env import TextWorldEnv
from utils.sb3_wrappers import TextWorldEncodingWrapper, TextWorldTrialEnv

# RL2 Imports
from agents.meta_rl_agent import RL2Agent
from utils.helpers import load_config
from meta_learning.rl2 import RL2

# RL2 Imports
from agents.meta_rl_agent import RL2Agent
from utils.helpers import load_config
from meta_learning.rl2 import RL2


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def visualize_agent_gameplay(
    model_path: str,
    game_path: str,
    config_path: Optional[str] = None,
    max_steps: int = 100,
    delay: float = 1.0,
    clear_between_steps: bool = False,
    save_transcript: Optional[str] = None
) -> Dict[str, Any]:
    """
    Watch the agent play a game step-by-step.
    
    Args:
        model_path: Path to the saved model (.zip for SB3, .pt for RL2)
        game_path: Path to the game file
        config_path: Path to config/meta_train.yaml (required for RL2)
        max_steps: Maximum steps per episode
        delay: Seconds to wait between steps
        clear_between_steps: Clear screen between steps
        save_transcript: Optional path to save transcript
        
    Returns:
        Episode summary (reward, steps, won/lost)
    """
    # Detect model type
    is_rl2 = model_path.endswith(".pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if is_rl2:
        if not config_path:
            raise ValueError("Config path is required for RL2 models (.pt)")
            
        print(f"{Colors.CYAN}Loading RL2 model...{Colors.ENDC}")
        config = load_config(config_path)
        
        # Reconstruct RL2 Agent
        agent_config = config.get("agent", {})
        distilbert_config = agent_config.get("distilbert", {})
        encoder_config = {
            "model_name": distilbert_config.get("model_name", "distilbert-base-uncased"),
            "freeze_layers": distilbert_config.get("freeze_layers", 4),
            "max_length": 512,
            "hidden_size": distilbert_config.get("hidden_size", 768),
        }
        
        policy_config = agent_config.get("policy", {})
        value_config = agent_config.get("value", {})
        rl2_config = config.get("rl2", {})
        
        agent = RL2Agent(
            encoder_config=encoder_config,
            policy_hidden_sizes=policy_config.get("hidden_sizes", [256, 128]),
            value_hidden_sizes=value_config.get("hidden_sizes", [256, 128]),
            hidden_size=encoder_config.get("hidden_size", 768),
            rnn_hidden_size=rl2_config.get("hidden_size", 256),
            device=device
        )
        
        # Load state dict
        try:
            # Try loading as complete RL2 checkpoint (including optimizer)
            checkpoint = torch.load(model_path, map_location=device)
            if "agent_state" in checkpoint:
                agent.load_state_dict(checkpoint["agent_state"])
            else:
                # Try direct state dict
                agent.load_state_dict(checkpoint)
        except Exception as e:
            print(f"{Colors.RED}Error loading checkpoint: {e}{Colors.ENDC}")
            return
            
        agent.eval()
        agent.to(device)
        agent.reset_hidden()
        
        # RL2 uses raw env
        env = TextWorldEnv(
            game_path=game_path,
            max_steps=max_steps,
            use_admissible_commands=True
        )
        # No extra wrappers for RL2 as it handles encoding internally
        
    else:
        # SB3 PPO
        print(f"{Colors.CYAN}Loading SB3 model...{Colors.ENDC}")
        model = PPO.load(model_path)
        
        # Create environment
        print(f"{Colors.CYAN}Loading game: {Path(game_path).name}{Colors.ENDC}")
        
        # We need the raw TextWorldEnv to access admissible commands
        raw_env = TextWorldEnv(
            game_path=game_path,
            max_steps=max_steps,
            use_admissible_commands=True
        )
        
        # Wrapped env for model
        env = TextWorldEncodingWrapper(raw_env, device=device)
        # Use TrialEnv wrapper to match training obs space (adds prev_action, etc.)
        env = TextWorldTrialEnv(env, episodes_per_trial=1)
    
    # Transcript storage
    transcript = []
    
    # Reset
    obs, info = env.reset()
    episode_reward = 0.0
    step_num = 0
    done = False
    won = False
    lost = False
    
    # Store raw env reference for accessing infos/commands
    if is_rl2:
        raw_env_ref = env
    else:
        # For SB3, env is wrapped
        raw_env_ref = env.unwrapped if hasattr(env, "unwrapped") else env
        # Actually our wrappers might hide .unwrapped properly, but let's try to get the base TextWorldEnv
        # In the SB3 block above, we created raw_env separately but then wrapped it into 'env'.
        # However, 'env' is what we step. 'raw_env' variable above is local to that block.
        # We need a unified way to get infos.
        # TextWorldTrialEnv wraps TextWorldEncodingWrapper which wraps TextWorldEnv.
        # Accessing .env.env should work
        raw_env_ref = env.env.env

    
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}GAME START: {Path(game_path).stem}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")
    
    time.sleep(delay)
    
    while not done:
        step_num += 1
        
        if clear_between_steps:
            clear_screen()
        
        # Get admissible commands
        if is_rl2:
             # raw_env_ref is TextWorldEnv, info contains admissible commands usually
             # But TextWorldEnv.step returns info with 'admissible_commands'
             # On reset/start, we might need to grab them from info or env
             admissible_commands = info.get("admissible_commands", ["look"])
        else:
            # SB3 wrappers might strip them or we use the side-channel raw_env (which is hard to sync)
            # Actually TextWorldEnv stores them in _current_infos
             admissible_commands = raw_env_ref.get_admissible_commands()
        
        # Get game state
        if is_rl2:
            description = info.get("description", "No description")
            inventory = info.get("inventory", "Nothing")
            feedback = info.get("feedback", "")
        else:
             description = raw_env_ref._current_infos.get("description", "No description")
             inventory = raw_env_ref._current_infos.get("inventory", "Nothing")
             feedback = raw_env_ref._current_obs if raw_env_ref._current_obs else ""
        
        # Get action
        if is_rl2:
             with torch.no_grad():
                action_idx, log_prob, value = agent.select_action(
                    obs, admissible_commands, deterministic=True
                )
        else:
            # SB3 PPO
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle numpy action
            if isinstance(action, np.ndarray):
                action_idx = int(action.item()) if action.size == 1 else int(action[0])
            else:
                action_idx = int(action)
        
        # Get the command string
        if action_idx < len(admissible_commands):
            chosen_command = admissible_commands[action_idx]
        else:
            chosen_command = admissible_commands[0] if admissible_commands else "look"
        
        # Display step header
        print(f"\n{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.YELLOW}STEP {step_num}{Colors.ENDC} | Cumulative Reward: {episode_reward:.2f}")
        print(f"{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
        
        # Display game state
        print(f"\n{Colors.BLUE}📍 LOCATION:{Colors.ENDC}")
        print(f"   {description[:300]}{'...' if len(description) > 300 else ''}")
        
        print(f"\n{Colors.BLUE}🎒 INVENTORY:{Colors.ENDC}")
        print(f"   {inventory}")
        
        if feedback and step_num > 1:
            print(f"\n{Colors.BLUE}💬 LAST FEEDBACK:{Colors.ENDC}")
            print(f"   {feedback[:200]}{'...' if len(feedback) > 200 else ''}")
        
        # Display admissible commands
        print(f"\n{Colors.CYAN}📋 ADMISSIBLE COMMANDS ({len(admissible_commands)}):{Colors.ENDC}")
        for i, cmd in enumerate(admissible_commands):
            if i == action_idx:
                print(f"   {Colors.GREEN}→ [{i}] {cmd} ← CHOSEN{Colors.ENDC}")
            else:
                print(f"     [{i}] {cmd}")
        
        # Store in transcript
        transcript.append({
            "step": step_num,
            "description": description,
            "inventory": inventory,
            "feedback": feedback,
            "admissible_commands": admissible_commands,
            "chosen_action": chosen_command,
            "action_idx": action_idx,
            "reward_before": episode_reward
        })
        
        # Take action
        if is_rl2:
             # RL2 agent uses raw action index directly to step TextWorldEnv
             # But we need to handle return values which are slightly different in raw env vs wrapped
             # TextWorldEnv: obs, reward, terminated, truncated, info
             obs, reward, terminated, truncated, info = env.step(action_idx)
             agent.update_prev_reward(reward)
        else:
             obs, reward, terminated, truncated, info = env.step(action)
             
        episode_reward += reward
        done = terminated or truncated
        
        # Display action result
        print(f"\n{Colors.YELLOW}⚡ ACTION: {chosen_command}{Colors.ENDC}")
        print(f"   Reward: {'+' if reward >= 0 else ''}{reward:.2f}")
        
        if info.get("won", False):
            won = True
            print(f"\n{Colors.GREEN}🎉 VICTORY!{Colors.ENDC}")
        if info.get("lost", False):
            lost = True
            print(f"\n{Colors.RED}💀 DEFEAT!{Colors.ENDC}")
        
        transcript[-1]["reward_after"] = episode_reward
        transcript[-1]["reward_step"] = reward
        
        if not done:
            time.sleep(delay)
    
    # Final summary
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}GAME OVER{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    
    status = "WON" if won else ("LOST" if lost else "TIMEOUT")
    status_color = Colors.GREEN if won else (Colors.RED if lost else Colors.YELLOW)
    
    print(f"\n   Status: {status_color}{status}{Colors.ENDC}")
    print(f"   Total Steps: {step_num}")
    print(f"   Total Reward: {episode_reward:.2f}")
    print(f"   Game Score: {info.get('score', 0)} / {info.get('max_score', 1)}")
    
    # Save transcript if requested
    if save_transcript:
        import json
        with open(save_transcript, 'w') as f:
            json.dump({
                "game": game_path,
                "model": model_path,
                "summary": {
                    "status": status,
                    "steps": step_num,
                    "reward": episode_reward,
                    "won": won,
                    "lost": lost
                },
                "transcript": transcript
            }, f, indent=2)
        print(f"\n   Transcript saved to: {save_transcript}")
    
    env.close()
    
    return {
        "status": status,
        "steps": step_num,
        "reward": episode_reward,
        "won": won,
        "lost": lost
    }


def main():
    parser = argparse.ArgumentParser(description="Watch SB3 agent play a TextWorld game")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SB3 model (.zip)")
    parser.add_argument("--game", type=str, required=True,
                        help="Path to game file (.z8 or .ulx)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode (default: 100)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between steps in seconds (default: 1.0)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear screen between steps")
    parser.add_argument("--save", type=str, default=None,
                        help="Save transcript to JSON file")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (required for RL2)")
    
    args = parser.parse_args()
    
    visualize_agent_gameplay(
        model_path=args.model,
        game_path=args.game,
        config_path=args.config,
        max_steps=args.max_steps,
        delay=args.delay,
        clear_between_steps=args.clear,
        save_transcript=args.save
    )


if __name__ == "__main__":
    main()
