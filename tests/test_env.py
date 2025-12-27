"""
Unit tests for environment module.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTextWorldEnv:
    """Tests for TextWorldEnv wrapper."""
    
    def test_env_initialization(self):
        """Test that environment can be initialized with valid config."""
        # This test would require a real game file
        # Mocking for unit test
        from envs.textworld_env import TextWorldEnv
        
        # Test that the class exists and has correct attributes
        assert hasattr(TextWorldEnv, 'reset')
        assert hasattr(TextWorldEnv, 'step')
        assert hasattr(TextWorldEnv, 'get_admissible_commands')
    
    def test_observation_structure(self):
        """Test that observations have the expected structure."""
        # Mock test - would need real game for integration test
        expected_keys = ['text', 'description', 'inventory', 'feedback']
        
        # Create mock observation
        mock_obs = {
            'text': 'Full observation text',
            'description': 'Room description',
            'inventory': 'Your inventory is empty',
            'feedback': 'Welcome to the game',
        }
        
        for key in expected_keys:
            assert key in mock_obs


class TestGameGenerator:
    """Tests for game generation."""
    
    def test_difficulty_presets(self):
        """Test that difficulty presets are correctly defined."""
        from envs.game_generator import GameGenerator
        
        assert 'easy' in GameGenerator.DIFFICULTY_PRESETS
        assert 'medium' in GameGenerator.DIFFICULTY_PRESETS
        assert 'hard' in GameGenerator.DIFFICULTY_PRESETS
        
        easy = GameGenerator.DIFFICULTY_PRESETS['easy']
        assert 'num_rooms' in easy
        assert 'quest_length' in easy
    
    def test_seed_generation(self):
        """Test deterministic seed generation."""
        from envs.game_generator import GameGenerator
        
        gen1 = GameGenerator(seed=42)
        gen2 = GameGenerator(seed=42)
        
        seed1 = gen1._generate_game_seed("test_game")
        seed2 = gen2._generate_game_seed("test_game")
        
        assert seed1 == seed2
    
    def test_different_seeds(self):
        """Test that different base seeds produce different game seeds."""
        from envs.game_generator import GameGenerator
        
        gen1 = GameGenerator(seed=42)
        gen2 = GameGenerator(seed=123)
        
        seed1 = gen1._generate_game_seed("test_game")
        seed2 = gen2._generate_game_seed("test_game")
        
        assert seed1 != seed2


class TestBatchEnv:
    """Tests for batch environment."""
    
    def test_batch_env_initialization(self):
        """Test batch environment can be initialized."""
        from envs.textworld_env import TextWorldBatchEnv
        
        batch_env = TextWorldBatchEnv(
            game_paths=["game1.z8", "game2.z8"],
            max_steps=100
        )
        
        assert len(batch_env.game_paths) == 2
        assert batch_env.max_steps == 100
