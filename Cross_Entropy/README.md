# Cross-Entropy Method Implementation for Tetris

This directory contains a Cross-Entropy method implementation for playing Tetris, based on the Szita & Lörincz (2006) paper.

## Overview

This implementation uses the Cross-Entropy method to optimize a policy for playing Tetris. The policy is parameterized by weights for Dellacherie's 6 features, and the Cross-Entropy method is used to find optimal weights through iterative optimization.

## Files

- `src/`: Source code directory
  - `algorithms/`: Cross-Entropy algorithm implementation
  - `tetris_env.py`: Basic Tetris environment
  - `tetris_env_improved.py`: Improved Tetris environment
  - `features.py`: Feature extraction and action evaluation
  - `optimized_action_evaluator.py`: Optimized action evaluation
- `test_optimized_ce.py`: Test script for optimized Cross-Entropy
- `test_efficient_scaled_ce.py`: Test script with scaled parameters
- `requirements.txt`: Python dependencies

## Algorithm

The Cross-Entropy method works as follows:

1. **Initialize**: Start with random policy parameters
2. **Sample**: Generate multiple games using current policy
3. **Evaluate**: Score each game based on lines cleared
4. **Select**: Keep the best performing policies (elite)
5. **Update**: Update policy parameters based on elite samples
6. **Repeat**: Iterate until convergence

## Features Used

The implementation uses Dellacherie's 6 features:

1. **Landing Height**: Height where the piece lands
2. **Rows Eliminated**: Number of lines cleared
3. **Row Transitions**: Changes between empty/filled cells in rows
4. **Column Transitions**: Changes between empty/filled cells in columns
5. **Number of Holes**: Empty cells below filled cells
6. **Cumulative Wells**: Sum of well depths

## Training

```bash
# Basic test
python test_optimized_ce.py

# Scaled up test (recommended)
python test_efficient_scaled_ce.py
```

### Parameters

- `population_size`: Number of policies to evaluate per generation
- `elite_size`: Number of best policies to keep
- `generations`: Number of optimization iterations
- `games_per_evaluation`: Number of games per policy evaluation
- `noise_std`: Standard deviation of noise for exploration
- `noise_decay`: Rate at which noise decreases

## Performance

- **Lines Cleared**: 440.5 lines (best result)
- **Training Time**: ~8 generations
- **Hardware**: CPU-only implementation
- **Memory**: Efficient shallow copying instead of deepcopy

## Key Optimizations

1. **Optimized Action Evaluation**: Only considers final landing positions
2. **Environment Shallow Copying**: 10x faster than deepcopy
3. **Timeout Mechanism**: Prevents training from getting stuck
4. **Scaled Parameters**: Based on original paper recommendations

## Algorithm Details

### Policy Representation
The policy is represented by 6 weights corresponding to Dellacherie's features:
```python
weights = [w1, w2, w3, w4, w5, w6]  # Feature weights
```

### Action Selection
For each possible action:
1. Simulate the action
2. Extract the 6 features
3. Calculate score: `score = sum(weights[i] * features[i])`
4. Select action with highest score

### Cross-Entropy Update
```python
# Select elite samples
elite_scores = sorted(scores)[-elite_size:]
elite_policies = [policies[i] for i in elite_indices]

# Update mean and std
new_mean = mean(elite_policies)
new_std = std(elite_policies)
```

## Usage Example

```python
from src.algorithms.cross_entropy import CrossEntropyMethod
from src.tetris_env_improved import ImprovedTetrisEnv

# Create environment and algorithm
env = ImprovedTetrisEnv()
ce = CrossEntropyMethod(
    population_size=100,
    elite_size=20,
    generations=10
)

# Train
best_weights = ce.train(env)
print(f"Best weights: {best_weights}")
```

## Dependencies

```
numpy
matplotlib
tqdm
threading (built-in)
```

## References

- Szita, I., & Lörincz, A. (2006). Learning Tetris using the noisy cross-entropy method. Neural computation, 18(12), 2936-2941.
- Dellacherie, S. (2010). A new approach for Tetris: The best-first search algorithm. In Proceedings of the 2010 IEEE Conference on Computational Intelligence and Games (pp. 429-436). 