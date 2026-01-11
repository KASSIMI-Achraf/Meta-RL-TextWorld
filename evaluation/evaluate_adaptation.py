"""
Adaptation Evaluation Pipeline

Comprehensive evaluation of meta-learning adaptation performance:
- Evaluate on held-out test games
- Compare against baselines
- Generate adaptation curves
- Produce summary statistics and visualizations
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.textworld_env import TextWorldEnv
from agents.meta_rl_agent import MetaRLAgent, RL2Agent
from agents.base_agent import RandomAgent
from meta_learning.rl2 import RL2
from meta_learning.inner_loop import collect_trajectories
from .metrics import (
    compute_metrics,
    compute_adaptation_curve,
    compute_adaptation_improvement,
    aggregate_task_results,
    EvaluationResult
)
from utils.helpers import set_seed, get_device, load_config


class AdaptationEvaluator:
    """
    Evaluates meta-learning adaptation performance on test games.
    
    Compares meta-learned agents against baselines and generates
    comprehensive evaluation reports.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/eval.yaml",
        algorithm: str = "maml",
        device: str = "auto"
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to evaluation config
            algorithm: Algorithm type
            device: Device to use
        """
        self.checkpoint_path = checkpoint_path
        self.config = load_config(config_path)
        self.algorithm_name = algorithm
        self.device = get_device(device)
        
        self.agent = None
        self.meta_learner = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if self.algorithm_name == "rl2":
            self.agent = RL2Agent(device=str(self.device))
            self.meta_learner = RL2.load(
                self.checkpoint_path,
                self.agent,
                device=str(self.device)
            )
        else:
            self.agent = MetaRLAgent(device=str(self.device))
            self.meta_learner = MAML.load(
                self.checkpoint_path,
                self.agent,
                device=str(self.device)
            )
        
        self.agent.to(self.device)
        self.agent.eval()
    
    def evaluate_single_game(
        self,
        game_path: str,
        adaptation_steps: List[int] = [0, 1, 2, 5, 10, 20],
        num_eval_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate adaptation on a single game.
        
        Args:
            game_path: Path to game file
            adaptation_steps: K values to evaluate
            num_eval_episodes: Evaluation episodes per K
            
        Returns:
            Evaluation results
        """
        env = TextWorldEnv(game_path, use_admissible_commands=True)
        
        results = {
            "game_path": game_path,
            "adaptation_curve": {},
            "final_metrics": None,
        }
        
        try:
            original_params = self.agent.clone_parameters()
            
            rewards_by_k = {}
            success_by_k = {}
            
            for k in adaptation_steps:
                # Reset to original parameters
                self.agent.load_parameters(original_params)
                
                if self.algorithm_name == "rl2":
                    self.agent.reset_hidden()
                
                # Adapt for k episodes
                if k > 0:
                    if self.algorithm_name == "rl2":
                        self.meta_learner.adapt(env, k)
                    else:
                        self.meta_learner.adapt(env, k)
                
                # Evaluate
                eval_trajectories = collect_trajectories(
                    self.agent, env, num_eval_episodes, deterministic=True
                )
                
                rewards = [t.total_reward() for t in eval_trajectories]
                success = [t.is_success() for t in eval_trajectories]
                
                rewards_by_k[k] = rewards
                success_by_k[k] = success
                
                results["adaptation_curve"][k] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "success_rate": np.mean(success),
                }
            
            # Final metrics (at max adaptation)
            max_k = max(adaptation_steps)
            final_rewards = rewards_by_k[max_k]
            final_success = success_by_k[max_k]
            
            results["final_metrics"] = compute_metrics(
                final_rewards, final_success
            ).to_dict()
            
            # Compute improvement
            if 0 in rewards_by_k and max_k in rewards_by_k:
                results["improvement"] = compute_adaptation_improvement(
                    rewards_by_k[0], rewards_by_k[max_k]
                )
            
            # Restore
            self.agent.load_parameters(original_params)
            
        finally:
            env.close()
        
        return results
    
    def evaluate_test_games(
        self,
        test_dir: str = "games/test",
        adaptation_steps: List[int] = [0, 1, 2, 5, 10, 20],
        num_eval_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate on all test games.
        
        Args:
            test_dir: Directory containing test games
            adaptation_steps: K values to evaluate
            num_eval_episodes: Evaluation episodes per K
            
        Returns:
            Aggregated results
        """
        test_path = Path(test_dir)
        game_files = list(test_path.glob("*.z8")) + list(test_path.glob("*.ulx"))
        
        if len(game_files) == 0:
            raise RuntimeError(f"No test games found in {test_dir}")
        
        print(f"Evaluating on {len(game_files)} test games...")
        
        all_results = []
        
        for game_path in tqdm(game_files, desc="Evaluating"):
            result = self.evaluate_single_game(
                str(game_path), adaptation_steps, num_eval_episodes
            )
            all_results.append(result)
        
        # Aggregate
        aggregated = self._aggregate_results(all_results, adaptation_steps)
        aggregated["individual_results"] = all_results
        
        return aggregated
    
    def _aggregate_results(
        self,
        results: List[Dict],
        adaptation_steps: List[int]
    ) -> Dict[str, Any]:
        """Aggregate results across games."""
        aggregated = {
            "num_games": len(results),
            "adaptation_curve": {},
            "overall_metrics": {},
        }
        
        for k in adaptation_steps:
            rewards_at_k = []
            success_at_k = []
            
            for r in results:
                if k in r["adaptation_curve"]:
                    rewards_at_k.append(r["adaptation_curve"][k]["mean_reward"])
                    success_at_k.append(r["adaptation_curve"][k]["success_rate"])
            
            if rewards_at_k:
                aggregated["adaptation_curve"][k] = {
                    "mean_reward": np.mean(rewards_at_k),
                    "std_reward": np.std(rewards_at_k),
                    "mean_success_rate": np.mean(success_at_k),
                }
        
        # Overall metrics at max adaptation
        max_k = max(adaptation_steps)
        if max_k in aggregated["adaptation_curve"]:
            aggregated["overall_metrics"] = aggregated["adaptation_curve"][max_k]
        
        return aggregated
    
    def compare_with_baselines(
        self,
        test_dir: str = "games/test",
        num_adaptation_episodes: int = 5,
        num_eval_episodes: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare meta-learned agent with baselines.
        
        Args:
            test_dir: Test games directory
            num_adaptation_episodes: Adaptation episodes
            num_eval_episodes: Evaluation episodes
            
        Returns:
            Comparison results
        """
        test_path = Path(test_dir)
        game_files = list(test_path.glob("*.z8")) + list(test_path.glob("*.ulx"))
        
        comparison = {
            "meta_learned": {"rewards": [], "success": []},
            "random": {"rewards": [], "success": []},
            "from_scratch": {"rewards": [], "success": []},
        }
        
        for game_path in tqdm(game_files, desc="Comparing"):
            # Meta-learned
            result = self.evaluate_single_game(
                str(game_path),
                adaptation_steps=[num_adaptation_episodes],
                num_eval_episodes=num_eval_episodes
            )
            comparison["meta_learned"]["rewards"].append(
                result["adaptation_curve"][num_adaptation_episodes]["mean_reward"]
            )
            comparison["meta_learned"]["success"].append(
                result["adaptation_curve"][num_adaptation_episodes]["success_rate"]
            )
            
            # Random
            env = TextWorldEnv(str(game_path), use_admissible_commands=True)
            random_agent = RandomAgent()
            random_trajs = collect_trajectories(random_agent, env, num_eval_episodes)
            comparison["random"]["rewards"].append(
                np.mean([t.total_reward() for t in random_trajs])
            )
            comparison["random"]["success"].append(
                np.mean([t.is_success() for t in random_trajs])
            )
            env.close()
        
        # Summarize
        summary = {}
        for method, data in comparison.items():
            summary[method] = {
                "mean_reward": np.mean(data["rewards"]),
                "std_reward": np.std(data["rewards"]),
                "mean_success": np.mean(data["success"]),
            }
        
        return summary
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate meta-learning adaptation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval.yaml",
                        help="Path to evaluation config")
    parser.add_argument("--test_dir", type=str, default="games/test",
                        help="Directory with test games")
    parser.add_argument("--algorithm", type=str, default="maml",
                        choices=["maml", "rl2"])
    parser.add_argument("--output", type=str, default="results/evaluation.json",
                        help="Output file for results")
    parser.add_argument("--compare_baselines", action="store_true",
                        help="Include baseline comparison")
    
    args = parser.parse_args()
    
    evaluator = AdaptationEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        algorithm=args.algorithm
    )
    
    # Run evaluation
    results = evaluator.evaluate_test_games(args.test_dir)
    
    if args.compare_baselines:
        results["baseline_comparison"] = evaluator.compare_with_baselines(args.test_dir)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Number of test games: {results['num_games']}")
    
    if "overall_metrics" in results:
        print(f"Final mean reward: {results['overall_metrics']['mean_reward']:.3f}")
        print(f"Final success rate: {results['overall_metrics']['mean_success_rate']:.2%}")


if __name__ == "__main__":
    main()
