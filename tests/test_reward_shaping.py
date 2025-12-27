import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock, patch
from envs.textworld_env import TextWorldEnv

class MockTextWorldEnv:
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.step_idx = 0
        
    def reset(self, seed=None):
        self.step_idx = 0
        return "Initial observation", self.scenarios[0]
        
    def step(self, command):
        self.step_idx += 1
        if self.step_idx >= len(self.scenarios):
            # Just repeat last state if out of bounds
            infos = self.scenarios[-1]
            done = True
        else:
            infos = self.scenarios[self.step_idx]
            done = False
            
        return "Observation", 0, done, infos
        
    def close(self):
        pass

@pytest.fixture
def mock_gym_make():
    with patch("gym.make") as mock_make:
        yield mock_make

@pytest.fixture
def mock_register():
    with patch("textworld.gym.register_game") as mock_reg:
        mock_reg.return_value = "tw-v0"
        yield mock_reg

def test_reward_shaping(mock_gym_make, mock_register):
    # Scenario definitions (List of infos for each step including reset)
    # Step 0: Reset
    # Step 1: Time pass (penalty only)
    # Step 2: New room (Exploration)
    # Step 3: Score up (Score gain)
    # Step 4: New item (Inventory)
    # Step 5: Win (Win bonus)
    
    scenarios = [
        # 0: Reset
        {
            "description": "Room A",
            "inventory": "Nothing",
            "score": 0,
            "won": False,
            "lost": False,
            "admissible_commands": ["look"]
        },
        # 1: Step 1 - Just time
        {
            "description": "Room A",
            "inventory": "Nothing",
            "score": 0,
            "won": False,
            "lost": False,
            "admissible_commands": ["look"]
        },
        # 2: Step 2 - New Room
        {
            "description": "Room B",
            "inventory": "Nothing",
            "score": 0,
            "won": False,
            "lost": False,
            "admissible_commands": ["look"]
        },
        # 3: Step 3 - Score Up
        {
            "description": "Room B",
            "inventory": "Nothing",
            "score": 5,
            "won": False,
            "lost": False,
            "admissible_commands": ["look"]
        },
        # 4: Step 4 - New Inventory
        {
            "description": "Room B",
            "inventory": "Sword",
            "score": 5,  # Score might stay same or increase, let's keep it same to isolate inventory reward if possible, 
                         # but usually taking item increases score. Let's assume independent for test.
            "won": False,
            "lost": False,
            "admissible_commands": ["look"]
        },
        # 5: Step 5 - Win
        {
            "description": "Room B",
            "inventory": "Sword",
            "score": 10,
            "won": True,
            "lost": False,
            "admissible_commands": ["look"]
        }
    ]
    
    # Configure shaping
    shaping_config = {
        "win_bonus": 10.0,
        "score_multiplier": 2.0,
        "exploration_bonus": 1.0,
        "inventory_bonus": 0.5,
        "time_penalty": -0.1,
        "productive_action": 0.0 # Ignore for this test
    }
    
    # Setup mock
    mock_env = MockTextWorldEnv(scenarios)
    mock_gym_make.return_value = mock_env
    
    # Initialize env
    env = TextWorldEnv(
        game_path="dummy.z8",
        reward_shaping=shaping_config
    )
    env.reset()
    
    # --- Step 1: Time Penalty ---
    _, reward, _, _, info = env.step("wait")
    # Expected: Time penalty only
    assert reward == pytest.approx(-0.1)
    
    # --- Step 2: Exploration ---
    _, reward, _, _, info = env.step("go north")
    # Expected: Time penalty + Exploration (Room B is new)
    assert reward == pytest.approx(-0.1 + 1.0)
    
    # --- Step 3: Score Gain ---
    _, reward, _, _, info = env.step("do something")
    # Expected: Time penalty + Score gain (5 - 0) * 2.0
    # Room is B (already visited), Inventory is Nothing (already seen)
    assert reward == pytest.approx(-0.1 + (5 * 2.0))
    
    # --- Step 4: Inventory ---
    _, reward, _, _, info = env.step("take sword")
    # Expected: Time penalty + Inventory (Sword is new)
    # Score is 5 (no change) in this scenario step
    assert reward == pytest.approx(-0.1 + 0.5)
    
    # --- Step 5: Win ---
    _, reward, _, _, info = env.step("win game")
    # Expected: Time penalty + Score gain (10 - 5) * 2.0 + Win bonus
    assert reward == pytest.approx(-0.1 + (5 * 2.0) + 10.0)

def test_revisit_no_reward(mock_gym_make, mock_register):
    # Test that revisiting states doesn't give reward
    scenarios = [
        # 0: Reset Room A
        {"description": "Room A", "inventory": "None", "score": 0, "won": False, "lost": False},
        # 1: Go Room B
        {"description": "Room B", "inventory": "None", "score": 0, "won": False, "lost": False},
        # 2: Go Back Room A
        {"description": "Room A", "inventory": "None", "score": 0, "won": False, "lost": False},
    ]
    
    mock_env = MockTextWorldEnv(scenarios)
    mock_gym_make.return_value = mock_env
    
    env = TextWorldEnv(
        game_path="dummy.z8",
        reward_shaping={
            "exploration_bonus": 1.0,
            "time_penalty": 0.0,
            "win_bonus": 0.0,
            "score_multiplier": 0.0,
            "inventory_bonus": 0.0,
            "productive_action": 0.0
        }
    )
    env.reset()
    
    # Step 1: New Room B
    _, reward, _, _, _ = env.step("go north")
    assert reward == 1.0
    
    # Step 2: Back to Room A (Already visited at reset)
    _, reward, _, _, _ = env.step("go south")
    assert reward == 0.0
