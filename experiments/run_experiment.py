"""
Experiment Runner

Main entry point for running meta-learning experiments.
Supports training, evaluation, and adaptation modes.

Usage:
    # Meta-training with MAML
    python experiments/run_experiment.py --config configs/meta_train.yaml --mode train --algorithm maml
    
    # Evaluate on test games
    python experiments/run_experiment.py --config configs/eval.yaml --mode eval --checkpoint checkpoints/best_model.pt
    
    # Adapt to a specific game
    python experiments/run_experiment.py --mode adapt --checkpoint checkpoints/best_model.pt --game games/test/treasure_hunter_test_0000.z8
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from training.meta_train import MetaTrainer
from training.adapt import Adapter
from evaluation.evaluate_adaptation import AdaptationEvaluator
from utils.helpers import set_seed, load_config

# SB3 Imports
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.sb3_wrappers import TextWorldTrialEnv, TextWorldEncodingWrapper
from agents.sb3_policy import TextWorldDistilBertPolicy
from envs.textworld_env import TextWorldEnv


def run_training(args):
    """Run meta-training."""
    print("=" * 60)
    print("META-TRAINING")
    print("=" * 60)
    
    if args.algorithm == "sb3":
        return run_sb3_training(args)
    
    trainer = MetaTrainer(
        config_path=args.config,
        algorithm=args.algorithm,
        seed=args.seed,
        debug=args.debug
    )
    
    history = trainer.run()
    
    # Save training history
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    history_path = results_dir / f"training_history_{args.algorithm}.json"
    with open(history_path, "w") as f:
        # Convert to serializable format (handle if history is None/empty)
        if history:
             serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
             json.dump(serializable, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    
    return history

def run_sb3_training(args):
    """Run training with Stable Baselines3 PPO."""
    print("Starting SB3 PPO Training...")
    
    config = load_config(args.config)
    
    from stable_baselines3.common.monitor import Monitor
    import os
    os.makedirs("logs/sb3_monitor", exist_ok=True)
    
    import random
    import glob
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.text_encoder import DistilBERTEncoder

    game_files = sorted(glob.glob(str(PROJECT_ROOT / "games/train/*.z8")))
    if not game_files:
        print(f"No games found in {PROJECT_ROOT / 'games/train'}, using fallback.")
        game_files = [str(PROJECT_ROOT / "games/train/train_0000.z8")] 
    
    print(f"Found {len(game_files)} training games.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    def make_sb3_env(rank, game_path):
        def _init():
            from envs.textworld_env import TextWorldEnv
            from utils.sb3_wrappers import TextWorldEncodingWrapper, TextWorldTrialEnv
            from stable_baselines3.common.monitor import Monitor
            
            env = TextWorldEnv(
                game_path=game_path,
                max_steps=100,
                use_admissible_commands=True
            )
            
            env = TextWorldEncodingWrapper(env, device=device)
            env = TextWorldTrialEnv(env, episodes_per_trial=10)
            
            log_dir = "logs/sb3_monitor"
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=f"{log_dir}/monitor_{rank}")
            return env
        return _init

    initial_game = game_files[0]
    print(f"Initializing model with {os.path.basename(initial_game)}...")
    env = DummyVecEnv([make_sb3_env(0, initial_game)])
    
    model = PPO(
        TextWorldDistilBertPolicy,
        env,
        verbose=1,
        learning_rate=config.get("meta_learning", {}).get("outer_lr", 1e-4),
        n_steps=128,
        batch_size=64,
        device=device,
        tensorboard_log=str(PROJECT_ROOT / "logs/sb3_ppo")
    )
    
    total_timesteps = 2000 if args.debug else config.get("meta_learning", {}).get("num_iterations", 1000) * 1280
    steps_per_game = 10000 
    num_updates = total_timesteps // steps_per_game
    
    print(f"Starting Sequential Training: {total_timesteps} steps total, switching game every {steps_per_game} steps.")
    
    current_timesteps = 0
    for i in range(num_updates):
        game_path = random.choice(game_files)
        
        new_env = DummyVecEnv([make_sb3_env(0, game_path)])
        
        model.set_env(new_env)
        
        model.learn(total_timesteps=steps_per_game, reset_num_timesteps=False, progress_bar=True)
        
        current_timesteps += steps_per_game
        if (i+1) % 10 == 0:
            print(f"Progress: {current_timesteps}/{total_timesteps} steps", flush=True)
            model.save(str(PROJECT_ROOT / "checkpoints/sb3_ppo_model_latest"))
    
    save_path = str(PROJECT_ROOT / "checkpoints/sb3_ppo_model")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    model.get_env().close()
    return model.get_env()


def run_evaluation(args):
    """Run evaluation on test games."""
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    evaluator = AdaptationEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        algorithm=args.algorithm
    )
    
    results = evaluator.evaluate_test_games(
        test_dir=args.test_dir or str(PROJECT_ROOT / "games/test")
    )
    
    if args.compare_baselines:
        results["baseline_comparison"] = evaluator.compare_with_baselines(
            test_dir=args.test_dir or str(PROJECT_ROOT / "games/test")
        )
    
    # Save results
    output_path = args.output or str(PROJECT_ROOT / "results/evaluation_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if "overall_metrics" in results:
        metrics = results["overall_metrics"]
        print(f"Mean Reward: {metrics.get('mean_reward', 0):.3f}")
        print(f"Success Rate: {metrics.get('mean_success_rate', 0):.2%}")
    
    if "baseline_comparison" in results:
        print("\nBaseline Comparison:")
        print("-" * 40)
        for method, m in results["baseline_comparison"].items():
            print(f"  {method:15s}: reward={m['mean_reward']:.3f}, "
                  f"success={m['mean_success']:.2%}")
    
    return results


def run_adaptation(args):
    """Run adaptation on a specific game."""
    print("=" * 60)
    print("ADAPTATION")
    print("=" * 60)
    
    adapter = Adapter(
        checkpoint_path=args.checkpoint,
        algorithm=args.algorithm
    )
    
    game_path = Path(args.game)
    
    if game_path.is_dir():
        # Adapt to all games in directory
        game_files = list(game_path.glob("*.z8")) + list(game_path.glob("*.ulx"))
        results = adapter.adapt_to_games(
            [str(g) for g in game_files],
            num_adaptation_episodes=args.adaptation_episodes,
            num_eval_episodes=args.eval_episodes
        )
    else:
        # Single game
        if args.compare_baselines:
            results = adapter.compare_with_baselines(
                str(game_path),
                num_adaptation_episodes=args.adaptation_episodes,
                num_eval_episodes=args.eval_episodes
            )
            
            print("\nComparison Results:")
            print("-" * 40)
            for method, m in results.items():
                print(f"  {method:15s}: reward={m['mean_reward']:.3f}, "
                      f"success={m['success_rate']:.2%}")
        else:
            results = adapter.adapt_to_game(
                str(game_path),
                num_adaptation_episodes=args.adaptation_episodes,
                num_eval_episodes=args.eval_episodes
            )
    
    return results


def run_experiment(args):
    """Main experiment dispatcher."""
    # Set seed
    set_seed(args.seed)
    
    if args.mode == "train":
        return run_training(args)
    elif args.mode == "eval":
        return run_evaluation(args)
    elif args.mode == "adapt":
        return run_adaptation(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Learning for TextWorld Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate games
  python envs/game_generator.py --num_train 100 --num_val 20 --num_test 30
  
  # Meta-train with MAML
  python experiments/run_experiment.py --mode train --algorithm maml --config configs/meta_train.yaml
  
  # Meta-train with RL²
  python experiments/run_experiment.py --mode train --algorithm rl2 --config configs/meta_train.yaml

  # Train with SB3 PPO (Baseline)
  python experiments/run_experiment.py --mode train --algorithm sb3
  
  # Evaluate on test games
  python experiments/run_experiment.py --mode eval --checkpoint checkpoints/best_model.pt
  
  # Adapt to a new game
  python experiments/run_experiment.py --mode adapt --game games/test/game.z8 --checkpoint checkpoints/best_model.pt
        """
    )
    
    # Mode and algorithm
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "adapt"],
                        help="Experiment mode")
    parser.add_argument("--algorithm", type=str, default="maml",
                        choices=["maml", "rl2", "sb3"],
                        help="Meta-learning algorithm")
    
    # Configuration
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/meta_train.yaml"),
                        help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (reduced iterations)")
    
    # Checkpoints
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for eval/adapt)")
    
    # Evaluation
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory with test games")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results")
    parser.add_argument("--compare_baselines", action="store_true",
                        help="Compare with baseline methods")
    
    # Adaptation
    parser.add_argument("--game", type=str, default=None,
                        help="Path to game file for adaptation")
    parser.add_argument("--adaptation_episodes", type=int, default=5,
                        help="Number of adaptation episodes")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["eval", "adapt"] and args.checkpoint is None:
        parser.error("--checkpoint is required for eval/adapt modes")
    
    if args.mode == "adapt" and args.game is None:
        parser.error("--game is required for adapt mode")
    
    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
