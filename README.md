# Meta-Learning for Fast Adaptation in TextWorld

A project evaluating meta-learning algorithms (RL²) for fast adaptation in procedurally generated text-based games using [TextWorld](https://github.com/microsoft/TextWorld).

## Question

**Can meta-learned agents adapt more quickly to unseen text-based games compared to agents trained from scratch?**

### Key Hypotheses
1. Meta-learned agents achieve higher rewards with fewer adaptation episodes
2. RL² outperforms random and from-scratch baselines in sample efficiency

## Project Structure

```
meta_textworld/
├── configs/                    # Configuration files
│   ├── env.yaml               # Game generation settings
│   ├── meta_train.yaml        # Meta-training hyperparameters
│   └── eval.yaml              # Evaluation settings
│
├── envs/                       # Environment wrappers
│   ├── textworld_env.py       # Gym-compatible TextWorld wrapper
│   └── game_generator.py      # Procedural game generation
│
├── agents/                     # Agent implementations
│   ├── base_agent.py          # Abstract base + random baseline
│   ├── text_encoder.py        # DistilBERT text encoder
│   └── meta_rl_agent.py       # RL² agent implementation
│
├── meta_learning/              # Meta-learning algorithms
│   ├── rl2.py                 # RL² implementation
│   ├── inner_loop.py          # Task adaptation
│   └── outer_loop.py          # Meta-optimization
│
├── training/                   # Training scripts
│   ├── meta_train.py          # Meta-training orchestrator
│   ├── adapt.py               # Fast adaptation
│   └── train_sb3_single.py    # SB3 baseline training
│
├── evaluation/                 # Evaluation pipeline
│   ├── evaluate_adaptation.py # Comprehensive evaluation
│   └── metrics.py             # Metric computation
│
├── utils/                      # Utilities
│   ├── replay_buffer.py       # Experience storage
│   ├── logger.py              # TensorBoard logging
│   └── helpers.py             # General utilities
│
├── experiments/                # Experiment runner
│   └── run_experiment.py      # Main entry point
│
├── games/                      # Generated games (created at runtime)
├── checkpoints/               # Model checkpoints (created at runtime)
├── logs/                      # Training logs (created at runtime)
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone or enter the project directory
cd Textworld-MetaRL

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `textworld>=1.5.0` - Text-based game environment
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.35.0` - DistilBERT encoder
- `gymnasium>=0.29.0` - RL environment interface
- `tensorboard>=2.14.0` - Training visualization
- `stable-baselines3>=2.0.0` - Standard RL baselines

## Docker Setup (Windows)

TextWorld requires Linux dependencies and doesn't run natively on Windows. Use Docker:

### Option 1: Use Existing TextWorld Container

If you have a TextWorld container (e.g., `my-textworld`):

```bash
# Start your container
docker start my-textworld

# Mount the project and get a shell
docker exec -it my-textworld /bin/bash

# Inside the container, navigate to mounted project and install deps
cd /path/to/mounted/project
pip install -r requirements.txt
```

### Option 2: Use Docker Compose

```bash
# Start the container with project mounted
docker-compose up -d

# Run commands inside
docker exec -it meta-textworld /bin/bash -c "cd /workspace && python experiments/run_experiment.py --mode train"

# Or use the helper script
docker_run.bat python experiments/run_experiment.py --mode train
```

### Running Commands in Docker

All commands in this README should be run inside the Docker container:

```bash
# Enter the container
docker exec -it my-textworld /bin/bash

# Then run commands normally
python experiments/run_experiment.py --mode train --algorithm rl2
```

## Quick Start

### 1. Generate Games

```bash
# Generate train/val/test game splits
python envs/game_generator.py --num_train 100 --num_val 20 --num_test 30 --difficulty easy
```

This creates:
- 100 training games in `games/train/`
- 20 validation games in `games/val/`
- 30 test games in `games/test/`

### 2. Meta-Training

```bash
# Train with RL² (context-based meta-learning)
python experiments/run_experiment.py --mode train --algorithm rl2 --config configs/meta_train.yaml

# Train standard SB3 Baseline
python experiments/run_experiment.py --mode train --algorithm sb3 --game games/train/train_0000.z8
```

For quick testing:
```bash
python experiments/run_experiment.py --mode train --algorithm rl2 --debug
```

### 3. Evaluate Adaptation

```bash
# Evaluate on test games
python experiments/run_experiment.py \
    --mode eval \
    --checkpoint checkpoints/best_model.pt \
    --algorithm rl2 \
    --compare_baselines

# Results saved to results/evaluation_results.json
```

### 4. Adapt to a New Game

```bash
# Adapt to a specific game
python experiments/run_experiment.py \
    --mode adapt \
    --checkpoint checkpoints/best_model.pt \
    --algorithm rl2 \
    --game games/test/treasure_hunter_test_0000.z8 \
    --adaptation_episodes 5 \
    --compare_baselines
```

## Algorithms

### RL² (Learning to Reinforcement Learn)

RL² uses a recurrent policy where the hidden state implicitly encodes task information.

**Key components:**
- **Recurrent policy**: GRU maintains hidden state across episodes
- **Context**: Hidden state accumulates task-specific information
- **Adaptation**: No gradient steps needed; adaptation through hidden state

```python
# Pseudocode
for task in tasks:
    hidden = init_hidden()
    for episode in range(episodes_per_trial):
        for step in episode:
            action, hidden = policy(obs, prev_action, prev_reward, hidden)
            # Hidden state learns to encode task information
```

## Agent Architecture

### Text Encoding (DistilBERT/TinyBERT)

The agent uses transformer-based models to encode text observations:

```
Observation: "You are in a kitchen. There is a table..."
     │
     ▼
┌─────────────────┐
│   Transformer   │  (frozen layers support)
│  Tokenizer +    │
│  Encoder        │
└────────┬────────┘
         │
         ▼
   [CLS] embedding
```

### Action Selection

Commands are scored using dot-product attention:

```
Observation Encoding ──┐
                       ├──► Dot Product ──► Softmax ──► Action Probabilities
Command Encodings ─────┘
```

## Configuration

### `configs/meta_train.yaml`

Key hyperparameters:

```yaml
meta_learning:
  num_iterations: 1000       # Meta-training iterations
  meta_batch_size: 4         # Tasks per batch
  outer_lr: 0.001            # Meta-optimizer learning rate
  
rl2:
  hidden_size: 256
  episodes_per_trial: 5

agent:
  encoder_type: "tinybert"
  tinybert:
    model_name: "huawei-noah/TinyBERT_General_4L_312D"
    freeze_layers: 2
```

## Evaluation Metrics

1. **Success Rate**: Proportion of games won after adaptation
2. **Mean Reward**: Average reward achieved after K adaptation episodes
3. **Adaptation Curve**: Reward vs. adaptation episodes (K = 0, 1, 2, 5, 10, 20)
4. **Sample Efficiency**: Episodes needed to reach performance threshold

## Task Structure

Each TextWorld game is a **task**:

| Split | Purpose | Default Count |
|-------|---------|---------------|
| Train | Meta-training | 100 games |
| Val | Hyperparameter tuning | 20 games |
| Test | Final evaluation | 30 games |

## Related Work

- **RL²**: Duan et al., "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning" (arXiv 2016)
- **TextWorld**: Côté et al., "TextWorld: A Learning Environment for Text-based Games" (CGW 2018)
- **KG-A2C**: Ammanabrolu & Hausknecht, "Graph Constrained Reinforcement Learning for Natural Language Action Spaces" (ICLR 2020)

## License

MIT License

## Citation

```bibtex
@misc{meta_textworld,
  title={Meta-Learning for Fast Adaptation in TextWorld},
  author={KASSIMI Achraf},
  year={2025},
  url={https://github.com/KASSIMI-Achraf/Meta-RL-TextWorld}
}
```
