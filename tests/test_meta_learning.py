"""
Unit tests for meta-learning module.
"""

import pytest
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTrajectory:
    """Tests for Trajectory dataclass."""
    
    def test_trajectory_creation(self):
        """Test trajectory can be created."""
        from meta_learning.inner_loop import Trajectory
        
        traj = Trajectory()
        
        assert len(traj) == 0
        assert traj.total_reward() == 0.0
        assert not traj.is_success()
    
    def test_trajectory_with_data(self):
        """Test trajectory with actual data."""
        from meta_learning.inner_loop import Trajectory
        
        traj = Trajectory(
            observations=[{'text': 'obs1'}, {'text': 'obs2'}],
            admissible_commands=[['go'], ['look']],
            actions=[0, 0],
            rewards=[0.5, 1.0],
            dones=[False, True],
            log_probs=[torch.tensor(0.0), torch.tensor(0.0)],
            values=[torch.tensor(0.0), torch.tensor(0.0)],
            infos=[{}, {'won': True}]
        )
        
        assert len(traj) == 2
        assert traj.total_reward() == 1.5
        assert traj.is_success()


class TestInnerLoop:
    """Tests for inner loop adaptation."""
    
    def test_inner_loop_initialization(self):
        """Test InnerLoop can be initialized."""
        from meta_learning.inner_loop import InnerLoop
        
        inner_loop = InnerLoop(
            inner_lr=0.01,
            num_steps=5,
            gamma=0.99
        )
        
        assert inner_loop.inner_lr == 0.01
        assert inner_loop.num_steps == 5
    
    def test_returns_computation(self):
        """Test GAE returns and advantages computation."""
        from meta_learning.inner_loop import InnerLoop, Trajectory
        
        inner_loop = InnerLoop(gamma=0.99, gae_lambda=0.95)
        
        traj = Trajectory(
            observations=[{}] * 3,
            admissible_commands=[[]] * 3,
            actions=[0, 0, 0],
            rewards=[0.0, 0.0, 1.0],
            dones=[False, False, True],
            log_probs=[torch.tensor(0.0)] * 3,
            values=[torch.tensor(0.0)] * 3,
            infos=[{}] * 3
        )
        
        returns, advantages = inner_loop.compute_returns_and_advantages(traj)
        
        assert len(returns) == 3
        assert len(advantages) == 3
        # Final reward should propagate backwards
        assert returns[-1].item() == 1.0


class TestMAML:
    """Tests for MAML algorithm."""
    
    def test_maml_initialization(self):
        """Test MAML can be initialized."""
        from meta_learning.maml import MAML
        from agents.meta_rl_agent import MetaRLAgent
        
        agent = MetaRLAgent(device="cpu")
        
        maml = MAML(
            agent=agent,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5
        )
        
        assert maml.inner_loop.inner_lr == 0.01
        assert maml.outer_loop.outer_lr == 0.001
    
    def test_maml_config(self):
        """Test MAML config storage."""
        from meta_learning.maml import MAML
        from agents.meta_rl_agent import MetaRLAgent
        
        agent = MetaRLAgent(device="cpu")
        
        maml = MAML(
            agent=agent,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=3,
            first_order=True
        )
        
        assert maml.config['inner_lr'] == 0.01
        assert maml.config['outer_lr'] == 0.001
        assert maml.config['first_order'] == True


class TestRL2:
    """Tests for RL² algorithm."""
    
    def test_rl2_initialization(self):
        """Test RL² can be initialized."""
        from meta_learning.rl2 import RL2
        from agents.meta_rl_agent import RL2Agent
        
        agent = RL2Agent(device="cpu")
        
        rl2 = RL2(
            agent=agent,
            learning_rate=0.001,
            episodes_per_trial=10
        )
        
        assert rl2.episodes_per_trial == 10
        assert rl2.learning_rate == 0.001
    
    def test_rl2_hidden_reset(self):
        """Test RL² agent hidden state reset."""
        from agents.meta_rl_agent import RL2Agent
        
        agent = RL2Agent(device="cpu")
        
        # Initial hidden should be None
        assert agent._hidden_state is None
        
        # After reset, should still be None
        agent.reset_hidden()
        assert agent._hidden_state is None


class TestOuterLoop:
    """Tests for outer loop meta-optimization."""
    
    def test_outer_loop_initialization(self):
        """Test OuterLoop can be initialized."""
        from meta_learning.outer_loop import OuterLoop
        from meta_learning.inner_loop import InnerLoop
        
        inner_loop = InnerLoop()
        outer_loop = OuterLoop(
            inner_loop=inner_loop,
            outer_lr=0.001,
            meta_batch_size=4
        )
        
        assert outer_loop.meta_batch_size == 4
        assert outer_loop.outer_lr == 0.001
    
    def test_task_sampling(self):
        """Test task batch sampling."""
        from meta_learning.outer_loop import OuterLoop
        from meta_learning.inner_loop import InnerLoop
        from unittest.mock import Mock
        
        inner_loop = InnerLoop()
        outer_loop = OuterLoop(
            inner_loop=inner_loop,
            meta_batch_size=2
        )
        
        # Create mock environments
        mock_envs = [Mock() for _ in range(5)]
        
        sampled = outer_loop.sample_task_batch(mock_envs)
        
        assert len(sampled) == 2
