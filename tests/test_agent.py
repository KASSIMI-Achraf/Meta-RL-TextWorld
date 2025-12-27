"""
Unit tests for agent module.
"""

import pytest
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""
    
    def test_random_agent_action_selection(self):
        """Test RandomAgent selects valid actions."""
        from agents.base_agent import RandomAgent
        
        agent = RandomAgent()
        
        obs = {'text': 'You are in a room.'}
        commands = ['go north', 'look', 'inventory']
        
        action_idx, log_prob, value = agent.select_action(obs, commands)
        
        assert 0 <= action_idx < len(commands)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
    
    def test_random_agent_empty_commands(self):
        """Test RandomAgent handles empty command list."""
        from agents.base_agent import RandomAgent
        
        agent = RandomAgent()
        
        obs = {'text': 'You are in a room.'}
        commands = []
        
        action_idx, log_prob, value = agent.select_action(obs, commands)
        
        assert action_idx == 0


class TestTextEncoder:
    """Tests for text encoders."""
    
    def test_distilbert_encoder_initialization(self):
        """Test DistilBERT encoder can be initialized."""
        from agents.text_encoder import DistilBERTEncoder
        
        encoder = DistilBERTEncoder(
            model_name="distilbert-base-uncased",
            freeze_layers=4,
            max_length=128
        )
        
        assert encoder.hidden_size == 768
        assert encoder.max_length == 128
    
    def test_distilbert_encode_single(self):
        """Test encoding a single text."""
        from agents.text_encoder import DistilBERTEncoder
        
        encoder = DistilBERTEncoder(max_length=64)
        
        text = "You are in a dark room."
        encoding = encoder.encode(text)
        
        assert encoding.shape == (768,)
    
    def test_distilbert_encode_batch(self):
        """Test encoding a batch of texts."""
        from agents.text_encoder import DistilBERTEncoder
        
        encoder = DistilBERTEncoder(max_length=64)
        
        texts = ["Room 1", "Room 2", "Room 3"]
        encodings = encoder.encode_batch(texts)
        
        assert encodings.shape == (3, 768)


class TestCommandScorer:
    """Tests for command scoring."""
    
    def test_command_scorer_output(self):
        """Test command scorer produces valid probabilities."""
        from agents.text_encoder import CommandScorer
        
        scorer = CommandScorer(hidden_size=768)
        
        obs_enc = torch.randn(768)
        cmd_enc = torch.randn(5, 768)  # 5 commands
        
        probs = scorer(obs_enc, cmd_enc)
        
        assert probs.shape == (5,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all()


class TestMetaRLAgent:
    """Tests for MetaRLAgent."""
    
    def test_agent_initialization(self):
        """Test MetaRLAgent can be initialized."""
        from agents.meta_rl_agent import MetaRLAgent
        
        agent = MetaRLAgent(device="cpu")
        
        assert hasattr(agent, 'encoder')
        assert hasattr(agent, 'policy')
        assert hasattr(agent, 'value')
    
    def test_agent_action_selection(self):
        """Test MetaRLAgent selects actions correctly."""
        from agents.meta_rl_agent import MetaRLAgent
        
        agent = MetaRLAgent(device="cpu")
        
        obs = {
            'text': 'You are in a kitchen.',
            'description': 'A small kitchen.',
            'inventory': 'Empty',
            'feedback': 'Welcome!'
        }
        commands = ['go north', 'look', 'take apple']
        
        action_idx, log_prob, value = agent.select_action(obs, commands)
        
        assert 0 <= action_idx < len(commands)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
    
    def test_agent_parameter_cloning(self):
        """Test parameter cloning and loading."""
        from agents.meta_rl_agent import MetaRLAgent
        
        agent = MetaRLAgent(device="cpu")
        
        # Clone parameters
        original_params = agent.clone_parameters()
        
        assert len(original_params) > 0
        
        # Modify a parameter
        for name, param in agent.named_parameters():
            param.data.add_(1.0)
            break
        
        # Load original
        agent.load_parameters(original_params)
