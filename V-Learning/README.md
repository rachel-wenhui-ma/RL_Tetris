# V-Learning Tetris AI Implementation

This is a V-Learning (Value Learning) implementation for Tetris based on Rex L's work, achieving state-of-the-art performance through neural network-based value function approximation.

## Performance Results

- **Best Score**: 1,470.1 lines cleared
- **Average Score**: 792 lines cleared  
- **Training Time**: 29.4 minutes (5 iterations)
- **Algorithm**: V-Learning with Convolutional Neural Networks

## Algorithm Overview

V-Learning is a value function approximation method that addresses the key challenge of Tetris: many steps are trivial for piece movement. The method operates at two levels:

1. **Learning Level**: Focuses on where to place pieces by directly evaluating final state values
2. **Execution Level**: Records action sequences to enable the AI to reach the chosen placement

### Key Features

- **V-learning approach**: Uses V(s') evaluation instead of Q(s,a) for more efficient learning
- **Environment assistance**: Game engine provides all possible piece placements
- **Search-based selection**: Enumerates all valid rotations, positions, and special moves
- **Neural architecture**: Uses CNN to learn state value functions

## Neural Network Architecture

### Main Grid Processing
- **CNN Branch A**: 64→32 filters with 6×6 and 3×3 kernels, max pooling (13,3)
- **CNN Branch B**: 128→32 filters with 4×4 and 3×3 kernels, max pooling (15,5)

### Input Structure
- **Main grid**: 20×10×1 binary matrix representing the board
- **Auxiliary features**: 46-dimensional vector (heights, holes, combo, piece information)
- **Total input size**: 246 dimensions

## Requirements

- Python 3.7+
- TensorFlow 2.5+
- NumPy
- Pygame

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Mode
```bash
python tetris_ai.py
```

### Configuration
Edit `common.py` to choose mode:
- `"ai_player_training"` - Training mode
- `"ai_player_watching"` - Watch trained AI play
- `"human_player"` - Human play mode

### Training Parameters
- **Buffer size**: 50,000 samples
- **Inner loop iterations**: 5
- **Epochs**: 5
- **Batch size**: 512
- **Optimizer**: Adam with learning rate 0.001
- **Loss function**: Mean squared error

## Training Process

The algorithm shows a characteristic pattern:
1. **Initial performance**: High initial performance (222.9 lines)
2. **Temporary dip**: Learning adjustment period (66.3 lines)
3. **Gradual recovery**: Steady improvement (152.0, 242.9 lines)
4. **Breakthrough**: Dramatic improvement (1,470.1 lines)

## Key Innovations

1. **Reduced complexity**: Dataset size reduced by approximately 1/7 (average steps to move a piece)
2. **Direct state evaluation**: AI chooses best placement location based on V-values
3. **Search-based action selection**: Enumerates all possible movement sequences
4. **Neural network generalization**: Leverages CNN for automatic feature learning

## References

- [Reinforcement Learning on Tetris - Medium Article](https://rex-l.medium.com/reinforcement-learning-on-tetris-707f75716c37)
- [Example Video](https://youtu.be/FTDZN4pPhwA)

## License

MIT License



    
