"""
TextWorld Game Generator (Simplified)

Generates TextWorld games with correct train/val/test splits.
Compatible with TextWorld 1.5+ and Docker.
"""

import hashlib
import json
import os
import click
import textworld
import textworld.generator
import textworld.challenges.tw_treasure_hunter.treasure_hunter
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Difficulty Presets
DIFFICULTY = {
    "easy": {"level": 1, "quest_length": 2, "num_rooms": 2},
    "medium": {"level": 10, "quest_length": 4, "num_rooms": 4},
    "hard": {"level": 20, "quest_length": 6, "num_rooms": 6},
}

def export_json(game: textworld.Game, output_path: str):
    # TextWorld's make_game returns a GameState-like object where infos dict holds entities
    # Keys starting with 'r_' are rooms
    
    data = {"rooms": {}, "player_start": "unknown"}

    if hasattr(game, "infos") and isinstance(game.infos, dict):
        # Iterate over rooms
        for key, entity_info in game.infos.items():
            if key.startswith("r_"):
                # It's a room
                room_name = entity_info.name if hasattr(entity_info, "name") else key
                
                # Exits and items might be harder to extract from this raw info
                # simplified export: just name and desc
                data["rooms"][room_name] = {
                    "description": entity_info.desc if hasattr(entity_info, "desc") else "No description",
                    "exits": {}, 
                    "items": [] 
                }
    
    # Try legacy attribute access if infos failed or yielded nothing
    if not data["rooms"] and hasattr(game, "rooms"):
         data["player_start"] = game.player.room.name if hasattr(game, "player") and game.player and game.player.room else "unknown"
         for room in game.rooms:
            items = []
            if hasattr(room.infos, "entities"):
                items = [{"name": e.name, "type": e.type, "desc": e.infos.desc} for e in room.infos.entities]
            exits = {d: r.name for d, r in room.exits.items()}
            data["rooms"][room.name] = {
                "description": room.infos.desc,
                "exits": exits,
                "items": items
            }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def generate_games(num_games, out_dir, difficulty="easy", seed_offset=0, prefix=""):
    """Generate a batch of games."""
    os.makedirs(out_dir, exist_ok=True)
    generated_files = []
    
    settings = DIFFICULTY.get(difficulty, DIFFICULTY["easy"])
    
    for i in range(num_games):
        game_id = f"{prefix}{i:04d}"
        seed = seed_offset + i
        
        # Set global seed for reproducibility
        textworld.g_rng.set_seed(seed)
        
        # 1. GENERATE
        options = textworld.GameOptions()
        options.seeds = seed
        
        # Treasure Hunter settings
        # Level 1 is simple, 30 is hard. We map difficulty to levels.
        # level determines rooms, objects, quest length internally for this challenge
        level = 1
        if difficulty == "medium": level = 10
        if difficulty == "hard": level = 20
        
        settings = {"level": level, "seed": seed}
        
        try:
            game = textworld.challenges.tw_treasure_hunter.treasure_hunter.make(settings, options)
        
            # 2. COMPILE
            game_file = os.path.join(out_dir, f"{game_id}.z8")
            options.path = game_file
            textworld.generator.compile_game(game, options=options)
            
            # 3. EXPORT JSON
            # Standard TextWorld metadata (for tw-view, tw-play --viewer)
            metadata_file = os.path.join(out_dir, f"{game_id}.json")
            game.save(metadata_file)

            # Custom JSON for GameMaker
            json_file = os.path.join(out_dir, f"{game_id}.vis.json")
            export_json(game, json_file)
            
            generated_files.append(game_file)

        except Exception as e:
            print(f"Error generating game {game_id}: {e}")
            continue
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_games} games in {out_dir}")

    return generated_files

@click.command()
@click.option("--num_train", default=10, help="Training games")
@click.option("--num_val", default=3, help="Validation games")
@click.option("--num_test", default=5, help="Test games")
@click.option("--output_dir", default="games", help="Output directory")
@click.option("--difficulty", default="easy", help="Difficulty level")
@click.option("--seed", default=42, help="Base random seed")
def main(num_train, num_val, num_test, output_dir, difficulty, seed):
    """Simple game generation script."""
    print(f"Generating games (Seed: {seed}, Difficulty: {difficulty})...")
    
    base_dir = Path(output_dir)
    
    # Train
    train_files = generate_games(num_train, base_dir / "train", difficulty, 
                                 seed_offset=seed, prefix="train_")
                                 
    # Val (offset seed to avoid overlap)
    val_files = generate_games(num_val, base_dir / "val", difficulty, 
                               seed_offset=seed + num_train, prefix="val_")
                               
    # Test (further offset)
    test_files = generate_games(num_test, base_dir / "test", difficulty, 
                                seed_offset=seed + num_train + num_val, prefix="test_")
                                
    # Save split info
    info = {
        "train": [str(p) for p in train_files],
        "val": [str(p) for p in val_files],
        "test": [str(p) for p in test_files],
        "settings": {"difficulty": difficulty, "seed": seed}
    }
    with open(base_dir / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)
        
    print(f"Done! Saved {len(train_files)+len(val_files)+len(test_files)} games to {output_dir}")

if __name__ == "__main__":
    main()