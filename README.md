# Tetris Reinforcement Learning Project

This project implements two different reinforcement learning approaches for playing Tetris:

1. **Q-Learning**: A tabular Q-learning implementation with state discretization
2. **Cross-Entropy Method**: A policy search implementation based on Szita & Lörincz (2006)

## Project Structure

```
RL_Tetris/
├── Q_Learning/           # Tabular Q-learning implementation
│   ├── src/             # Source code
│   ├── train.py         # Training script
│   ├── demo.py          # Demo script
│   ├── test_basic.py    # Basic tests
│   └── requirements.txt # Dependencies
├── Cross_Entropy/       # Cross-Entropy method implementation
│   ├── src/             # Source code
│   ├── test_optimized_ce.py      # Optimized CE test
│   ├── test_efficient_scaled_ce.py # Scaled CE test
│   └── requirements.txt # Dependencies
├── data/                # Training results and models
│   ├── models/          # Saved models
│   └── plots/           # Training plots
└── README.md           # This file
```

## Q-Learning Implementation

### Features
- Tabular Q-learning with state discretization
- Height profile and hole count state representation
- Gymnasium-compatible environment
- CPU-friendly implementation

### Usage
```bash
cd Q_Learning
pip install -r requirements.txt
python train.py
python demo.py
```

### State Space
- Height profile of the board
- Number of holes
- Current and next piece information

### Action Space
- 44 possible actions (4 rotations × 11 horizontal positions)

## Cross-Entropy Implementation

### Features
- Based on Szita & Lörincz (2006) paper
- Dellacherie's 6 features for state evaluation
- Optimized action evaluation (only final landing positions)
- CPU-friendly policy search method

### Usage
```bash
cd Cross_Entropy
pip install -r requirements.txt
python test_efficient_scaled_ce.py
```

### Performance
- Achieved 440.5 lines cleared in testing
- Training time: ~8 generations
- CPU-only implementation

### Features Used
1. **Landing Height**: Height where piece lands
2. **Rows Eliminated**: Number of lines cleared
3. **Row Transitions**: Changes between empty/filled cells in rows
4. **Column Transitions**: Changes between empty/filled cells in columns
5. **Number of Holes**: Empty cells below filled cells
6. **Cumulative Wells**: Sum of well depths

## Results Comparison

| Method | Lines Cleared | Training Time | Hardware |
|--------|---------------|---------------|----------|
| Q-Learning | ~10-20 | 5000 episodes | CPU |
| Cross-Entropy | 440.5 | 8 generations | CPU |

## Dependencies

### Q-Learning
- gymnasium
- numpy
- matplotlib
- tqdm

### Cross-Entropy
- numpy
- matplotlib
- tqdm
- threading (for timeout)

## Key Innovations

1. **State Discretization**: Different approaches for Q-learning vs Cross-Entropy
2. **Optimized Action Evaluation**: Only considering final landing positions in CE
3. **Environment Shallow Copying**: Significant performance improvement over deepcopy
4. **Timeout Mechanism**: Prevents training from getting stuck

## References

- Szita, I., & Lörincz, A. (2006). Learning Tetris using the noisy cross-entropy method. Neural computation, 18(12), 2936-2941.
- Dellacherie, S. (2010). A new approach for Tetris: The best-first search algorithm. In Proceedings of the 2010 IEEE Conference on Computational Intelligence and Games (pp. 429-436).

## License

MIT License 