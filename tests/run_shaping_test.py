
import sys
import os
import gymnasium as gym
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.getcwd())

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
            infos = self.scenarios[-1]
            done = True
        else:
            infos = self.scenarios[self.step_idx]
            done = False
            
        return "Observation", 0.0, done, infos
        
    def close(self):
        pass

def test_reward_shaping():
    print("Testing reward shaping...")
    
    scenarios = [
        # 0: Reset
        {"description": "Room A", "inventory": "Nothing", "score": 0, "won": False, "lost": False, "admissible_commands": ["look"]},
        # 1: Step 1 - Just time
        {"description": "Room A", "inventory": "Nothing", "score": 0, "won": False, "lost": False, "admissible_commands": ["look"]},
        # 2: Step 2 - New Room
        {"description": "Room B", "inventory": "Nothing", "score": 0, "won": False, "lost": False, "admissible_commands": ["look"]},
        # 3: Step 3 - Score Up
        {"description": "Room B", "inventory": "Nothing", "score": 5, "won": False, "lost": False, "admissible_commands": ["look"]},
        # 4: Step 4 - New Inventory
        {"description": "Room B", "inventory": "Sword", "score": 5, "won": False, "lost": False, "admissible_commands": ["look"]},
        # 5: Step 5 - Win
        {"description": "Room B", "inventory": "Sword", "score": 10, "won": True, "lost": False, "admissible_commands": ["look"]}
    ]
    
    shaping_config = {
        "win_bonus": 10.0,
        "score_multiplier": 2.0,
        "exploration_bonus": 1.0,
        "inventory_bonus": 0.5,
        "time_penalty": -0.1,
        "productive_action": 0.0
    }
    
    mock_env = MockTextWorldEnv(scenarios)
    
    with patch("gym.make", return_value=mock_env):
        with patch("textworld.gym.register_game", return_value="tw-v0"):
            env = TextWorldEnv(game_path="dummy.z8", reward_shaping=shaping_config)
            env.reset()
            
            # Step 1: Time
            _, reward, _, _, _ = env.step("wait")
            print(f"Step 1 Reward: {reward} (Expected: -0.1)")
            assert abs(reward - (-0.1)) < 1e-5
            
            # Step 2: Exploration
            _, reward, _, _, _ = env.step("go north")
            print(f"Step 2 Reward: {reward} (Expected: 0.9)")
            assert abs(reward - (-0.1 + 1.0)) < 1e-5
            
            # Step 3: Score
            _, reward, _, _, _ = env.step("do something")
            print(f"Step 3 Reward: {reward} (Expected: 9.9)")
            assert abs(reward - (-0.1 + 10.0)) < 1e-5
            
            # Step 4: Inventory
            _, reward, _, _, _ = env.step("take sword")
            print(f"Step 4 Reward: {reward} (Expected: 0.4)")
            assert abs(reward - (-0.1 + 0.5)) < 1e-5
            
            # Step 5: Win
            _, reward, _, _, _ = env.step("win")
            print(f"Step 5 Reward: {reward} (Expected: 19.9)")
            assert abs(reward - (-0.1 + 10.0 + 10.0)) < 1e-5
            
    print("Reward shaping test passed!")

if __name__ == "__main__":
    test_reward_shaping()
