"""
Evaluation Metrics

Computes various metrics for evaluating meta-learning performance:
- Success rate
- Sample efficiency
- Adaptation curves
- Statistical significance
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    mean_reward: float
    std_reward: float
    success_rate: float
    sample_efficiency: Optional[int]
    rewards_per_episode: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "success_rate": self.success_rate,
            "sample_efficiency": self.sample_efficiency,
            "rewards_per_episode": self.rewards_per_episode,
        }


def compute_success_rate(successes: List[bool]) -> float:
    """
    Compute success rate from list of success indicators.
    
    Args:
        successes: List of boolean success indicators
        
    Returns:
        Success rate as proportion
    """
    if len(successes) == 0:
        return 0.0
    return np.mean(successes)


def compute_sample_efficiency(
    rewards: List[float],
    threshold: float = 0.8,
    max_reward: float = 1.0
) -> Optional[int]:
    """
    Compute sample efficiency as episodes to reach reward threshold.
    
    Args:
        rewards: List of rewards per episode
        threshold: Threshold as proportion of max_reward to reach
        max_reward: Maximum possible reward
        
    Returns:
        Number of episodes to reach threshold, or None if not reached
    """
    target = threshold * max_reward
    
    for i, reward in enumerate(rewards):
        if reward >= target:
            return i + 1
    
    return None


def compute_adaptation_curve(
    rewards_by_episode: List[List[float]],
    adaptation_steps: List[int] = [0, 1, 2, 5, 10, 20]
) -> Dict[int, Dict[str, float]]:
    """
    Compute adaptation curve: reward vs. adaptation steps.
    
    Args:
        rewards_by_episode: List of reward lists, one per task
        adaptation_steps: Adaptation step values to evaluate
        
    Returns:
        Dictionary mapping adaptation steps to reward statistics
    """
    results = {}
    
    for k in adaptation_steps:
        # Get reward at step k for each task
        rewards_at_k = []
        for task_rewards in rewards_by_episode:
            if k < len(task_rewards):
                rewards_at_k.append(task_rewards[k])
            elif len(task_rewards) > 0:
                rewards_at_k.append(task_rewards[-1])
        
        if len(rewards_at_k) > 0:
            results[k] = {
                "mean": np.mean(rewards_at_k),
                "std": np.std(rewards_at_k),
                "min": np.min(rewards_at_k),
                "max": np.max(rewards_at_k),
            }
    
    return results


def compute_metrics(
    rewards: List[float],
    successes: List[bool],
    threshold: float = 0.8,
    max_reward: float = 1.0
) -> EvaluationResult:
    """
    Compute all evaluation metrics.
    
    Args:
        rewards: List of rewards per episode
        successes: List of success indicators
        threshold: Threshold for sample efficiency
        max_reward: Maximum possible reward
        
    Returns:
        EvaluationResult with all metrics
    """
    return EvaluationResult(
        mean_reward=np.mean(rewards) if rewards else 0.0,
        std_reward=np.std(rewards) if rewards else 0.0,
        success_rate=compute_success_rate(successes),
        sample_efficiency=compute_sample_efficiency(rewards, threshold, max_reward),
        rewards_per_episode=rewards,
    )


def compute_adaptation_improvement(
    pre_adaptation_rewards: List[float],
    post_adaptation_rewards: List[float]
) -> Dict[str, float]:
    """
    Compute improvement from adaptation.
    
    Args:
        pre_adaptation_rewards: Rewards before adaptation
        post_adaptation_rewards: Rewards after adaptation
        
    Returns:
        Improvement metrics
    """
    pre_mean = np.mean(pre_adaptation_rewards) if pre_adaptation_rewards else 0.0
    post_mean = np.mean(post_adaptation_rewards) if post_adaptation_rewards else 0.0
    
    absolute_improvement = post_mean - pre_mean
    relative_improvement = absolute_improvement / (abs(pre_mean) + 1e-8)
    
    return {
        "pre_adaptation_mean": pre_mean,
        "post_adaptation_mean": post_mean,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
    }


def statistical_comparison(
    method1_rewards: List[float],
    method2_rewards: List[float],
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical comparison between two methods.
    
    Uses Welch's t-test for comparing means.
    
    Args:
        method1_rewards: Rewards from method 1
        method2_rewards: Rewards from method 2
        significance_level: Alpha level for significance
        
    Returns:
        Statistical comparison results
    """
    from scipy import stats
    
    if len(method1_rewards) < 2 or len(method2_rewards) < 2:
        return {
            "error": "Not enough samples for statistical comparison"
        }
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(
        method1_rewards, 
        method2_rewards, 
        equal_var=False
    )
    
    significant = p_value < significance_level
    
    return {
        "method1_mean": np.mean(method1_rewards),
        "method2_mean": np.mean(method2_rewards),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": significant,
        "significance_level": significance_level,
    }


def aggregate_task_results(
    task_results: List[EvaluationResult]
) -> Dict[str, float]:
    """
    Aggregate results across multiple tasks.
    
    Args:
        task_results: List of per-task evaluation results
        
    Returns:
        Aggregated statistics
    """
    if len(task_results) == 0:
        return {}
    
    all_rewards = [r.mean_reward for r in task_results]
    all_success = [r.success_rate for r in task_results]
    
    efficiencies = [r.sample_efficiency for r in task_results if r.sample_efficiency is not None]
    
    return {
        "mean_reward_across_tasks": np.mean(all_rewards),
        "std_reward_across_tasks": np.std(all_rewards),
        "mean_success_rate": np.mean(all_success),
        "num_tasks_reaching_threshold": len(efficiencies),
        "mean_sample_efficiency": np.mean(efficiencies) if efficiencies else None,
        "num_tasks": len(task_results),
    }
