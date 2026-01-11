"""
Adaptation Script

Fast adaptation to new/unseen games using a pre-trained RL² meta-model.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.textworld_env import TextWorldEnv
from agents.meta_rl_agent import RL2Agent
from agents.base_agent import RandomAgent
from meta_learning.rl2 import RL2
from meta_learning.inner_loop import collect_trajectories
from utils.helpers import set_seed, get_device, load_config


class Adapter:
    """
    Handles fast adaptation of meta-trained RL² agents to new games.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto"
    ):
        """
        Initialize the adapter.
        
        Args:
            checkpoint_path: Path to meta-trained model checkpoint
            device: Device to run on
        """
        self.checkpoint_path = checkpoint_path
        self.device = get_device(device)
        
        self.agent = None
        self.meta_learner = None
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load the meta-trained checkpoint."""
        path = Path(self.checkpoint_path)
        
        # Handle directory input
        if path.is_dir():
            if (path / "best_model.pt").exists():
                path = path / "best_model.pt"
            else:
                pt_files = list(path.glob("*.pt"))
                if pt_files:
                    path = pt_files[0]
                    print(f"No best_model.pt found, using {path.name} instead.")
                else:
                    raise FileNotFoundError(f"No .pt checkpoint found in {path}")
            self.checkpoint_path = str(path)

        self.agent = RL2Agent(device=str(self.device))
        self.meta_learner = RL2.load(
            self.checkpoint_path,
            self.agent,
            device=str(self.device)
        )
        
        self.agent.to(self.device)
        print(f"Loaded RL2 checkpoint from {self.checkpoint_path}")
    
    def adapt_to_game(
        self,
        game_path: str,
        num_adaptation_episodes: int = 5,
        num_eval_episodes: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Adapt to a new game and evaluate.
        """
        env = TextWorldEnv(
            game_path=game_path,
            max_steps=100,
            use_admissible_commands=True
        )
        
        results = {
            "game_path": game_path,
            "num_adaptation_episodes": num_adaptation_episodes,
            "adaptation_rewards": [],
            "eval_rewards": [],
            "eval_success_rate": 0.0,
        }
        
        try:
            original_params = self.agent.clone_parameters()
            
            if verbose:
                print(f"\nAdapting to: {Path(game_path).name}")
                print(f"  Adaptation episodes: {num_adaptation_episodes}")
            
            # RL²: adaptation through hidden state
            adapt_trajectories = self.meta_learner.adapt(env, num_adaptation_episodes)
            results["adaptation_rewards"] = [t.total_reward() for t in adapt_trajectories]
            
            if verbose:
                mean_adapt_reward = np.mean(results["adaptation_rewards"])
                print(f"  Mean adaptation reward: {mean_adapt_reward:.3f}")
            
            eval_trajectories = collect_trajectories(
                self.agent, env, num_eval_episodes, deterministic=True
            )
            
            results["eval_rewards"] = [t.total_reward() for t in eval_trajectories]
            results["eval_success_rate"] = np.mean([t.is_success() for t in eval_trajectories])
            
            if verbose:
                print(f"  Eval mean reward: {np.mean(results['eval_rewards']):.3f}")
                print(f"  Eval success rate: {results['eval_success_rate']:.2%}")
            
            self.agent.load_parameters(original_params)
            
        finally:
            env.close()
        
        return results
    
    def adapt_to_games(
        self,
        game_paths: List[str],
        num_adaptation_episodes: int = 5,
        num_eval_episodes: int = 10
    ) -> List[Dict[str, Any]]:
        """Adapt to multiple games."""
        all_results = []
        
        for game_path in game_paths:
            result = self.adapt_to_game(
                game_path,
                num_adaptation_episodes,
                num_eval_episodes
            )
            all_results.append(result)
        
        return all_results
    
    def compare_with_baselines(
        self,
        game_path: str,
        num_adaptation_episodes: int = 5,
        num_eval_episodes: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Compare meta-learned agent with baselines."""
        results = {}
        
        meta_result = self.adapt_to_game(
            game_path,
            num_adaptation_episodes,
            num_eval_episodes,
            verbose=False
        )
        results["meta_learned"] = {
            "mean_reward": np.mean(meta_result["eval_rewards"]),
            "success_rate": meta_result["eval_success_rate"],
        }
        
        env = TextWorldEnv(game_path, use_admissible_commands=True)
        random_agent = RandomAgent()
        random_trajectories = collect_trajectories(random_agent, env, num_eval_episodes)
        results["random"] = {
            "mean_reward": np.mean([t.total_reward() for t in random_trajectories]),
            "success_rate": np.mean([t.is_success() for t in random_trajectories]),
        }
        env.close()
        
        return results


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapt meta-trained agent to new games")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to meta-trained checkpoint")
    parser.add_argument("--game", type=str, required=True,
                        help="Path to game file or directory")
    parser.add_argument("--adaptation_episodes", type=int, default=5,
                        help="Number of adaptation episodes")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--compare_baselines", action="store_true",
                        help="Compare with baseline methods")
    
    args = parser.parse_args()
    
    adapter = Adapter(checkpoint_path=args.checkpoint)
    
    game_path = Path(args.game)
    
    if game_path.is_dir():
        game_files = list(game_path.glob("*.z8")) + list(game_path.glob("*.ulx"))
        adapter.adapt_to_games(
            [str(g) for g in game_files],
            args.adaptation_episodes,
            args.eval_episodes
        )
    else:
        if args.compare_baselines:
            results = adapter.compare_with_baselines(
                str(game_path),
                args.adaptation_episodes,
                args.eval_episodes
            )
            print("\nComparison Results:")
            print("-" * 40)
            for method, metrics in results.items():
                print(f"{method:15s}: reward={metrics['mean_reward']:.3f}, "
                      f"success={metrics['success_rate']:.2%}")
        else:
            adapter.adapt_to_game(
                str(game_path),
                args.adaptation_episodes,
                args.eval_episodes
            )


if __name__ == "__main__":
    main()
