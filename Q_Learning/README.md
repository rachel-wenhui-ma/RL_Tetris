# Q-Learning Implementation for Tetris

This directory contains a tabular Q-learning implementation for playing Tetris.

## Overview

This implementation uses tabular Q-learning with state discretization to learn how to play Tetris. The state space is discretized using height profiles and hole counts to make the problem tractable for tabular methods.

## Files

- `src/`: Source code directory
  - `agents/`: Q-learning agent implementation
  - `environment/`: Tetris game environment
- `train.py`: Training script
- `demo.py`: Demo script to visualize trained agent
- `test_basic.py`: Basic tests for the implementation
- `requirements.txt`: Python dependencies

## State Space

The state is discretized using:
- **Height Profile**: The height of each column (10 values)
- **Hole Count**: Number of holes in the board
- **Current Piece**: Type of current tetromino (7 types)
- **Next Piece**: Type of next tetromino (7 types)

## Action Space

- **Rotations**: 4 possible rotations (0°, 90°, 180°, 270°)
- **Horizontal Positions**: 11 possible positions
- **Total Actions**: 44 possible actions per piece

## Training

```bash
python train.py --episodes 5000 --save_interval 1000
```

### Parameters
- `episodes`: Number of training episodes
- `save_interval`: How often to save the model
- `epsilon`: Exploration rate (starts at 1.0, decays to 0.01)
- `learning_rate`: Q-learning learning rate
- `discount_factor`: Future reward discount

## Demo

```bash
python demo.py
```

This will load a trained model and show the agent playing Tetris.

## Performance

- **Lines Cleared**: ~10-20 lines per game
- **Training Time**: ~5000 episodes
- **Hardware**: CPU-only implementation

## Key Features

1. **State Discretization**: Makes the infinite state space tractable
2. **Gymnasium Compatible**: Standard RL environment interface
3. **CPU Friendly**: No GPU requirements
4. **Modular Design**: Separate agent and environment components

## Dependencies

```
gymnasium
numpy
matplotlib
tqdm
```

## Usage Example

```python
from src.environment.tetris_env import TetrisEnv
from src.agents.q_learning_agent import QLearningAgent

# Create environment and agent
env = TetrisEnv()
agent = QLearningAgent(state_size=env.observation_space.shape[0])

# Train the agent
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
``` 