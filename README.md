# Tetris Reinforcement Learning Project

This project implements two main reinforcement learning approaches for playing Tetris:

1. **V-Learning**: A value-based learning approach with CNN architecture
2. **CEM (Cross-Entropy Method)**: A policy search implementation with paper-accurate parameters

## Project Introduction

This project explores reinforcement learning techniques for the classic puzzle game Tetris. We implemented and compared different approaches, from basic tabular methods to sophisticated neural network architectures. The project demonstrates the evolution of RL techniques and provides practical implementations for Tetris AI research.

Our main contributions include:
- A CNN-based V-Learning implementation achieving 1,470+ lines cleared
- A paper-accurate Cross-Entropy Method implementation following original research parameters
- Comprehensive training results and analysis
- Production-ready code with detailed documentation

## Project Structure

```
RL_Tetris/
├── V-Learning/          # Main implementation: Value-based learning with CNN
│   ├── tetris_ai.py     # Core V-Learning implementation
│   ├── game.py          # Tetris game environment
│   ├── requirements.txt # Dependencies
│   └── README.md        # Detailed V-Learning documentation
├── CEM/                 # Main implementation: Paper-accurate Cross-Entropy Method
│   ├── cem_paper_implementation.py # Core CEM implementation
│   ├── run_cem_paper.py          # Training script
│   ├── training_results/          # Training outputs and logs
│   ├── requirements.txt           # Dependencies
│   └── README_CEM_Paper.md       # Detailed CEM documentation
├── Q_Learning/          # Early exploration: Basic tabular Q-learning
│   ├── src/             # Source code
│   ├── train.py         # Training script
│   └── requirements.txt # Dependencies
├── Cross_Entropy/       # Early exploration: Basic Cross-Entropy method
│   ├── src/             # Source code
│   ├── test_optimized_ce.py      # Basic CE test
│   └── requirements.txt # Dependencies
└── README.md           # This file
```

## Main Implementations

### V-Learning (Value-Based Learning)
**Location**: `V-Learning/`

**Features**:
- CNN-based neural network architecture
- Value function approximation
- Action sequence generation and evaluation
- Advanced reward shaping with line clear bonuses

**Performance**:
- Best score: 1,470.1 lines cleared
- Average performance: 792 lines
- Training time: 29.4 minutes

**Usage**:
```bash
cd V-Learning
pip install -r requirements.txt
python tetris_ai.py
```

### CEM (Cross-Entropy Method)
**Location**: `CEM/`

**Features**:
- Strictly paper-accurate implementation
- Population size: 100 (paper standard)
- Elite ratio: 0.1 (paper standard)
- Games per evaluation: 30 (paper standard)
- Adaptive noise strategy: Zt = max(5 - t/10, 0)

**Performance**:
- Generation 1: Best=304.1, Mean=4.1
- Generation 2: Best=96.3, Mean=11.6
- Generation 3: Best=350.2, Mean=83.0
- Generation 4: Best=400.1, Mean=179.5

**Usage**:
```bash
cd CEM
pip install -r requirements.txt
python test_cem_paper.py  # Test implementation
python run_cem_paper.py    # Start training
```

## Early Exploratory Experiments

### Q-Learning (Tabular Approach)
**Location**: `Q_Learning/`

**Purpose**: Early exploration of tabular Q-learning for Tetris
**Features**: Basic state discretization, simple action space
**Performance**: Limited success (~10-20 lines)
**Status**: Superseded by V-Learning approach

### Cross-Entropy (Basic Implementation)
**Location**: `Cross_Entropy/`

**Purpose**: Early exploration of policy search methods
**Features**: Basic Dellacherie features, simplified parameters
**Performance**: Achieved 440.5 lines in testing
**Status**: Superseded by paper-accurate CEM implementation

## Dependencies

### V-Learning
- tensorflow
- numpy
- matplotlib

### CEM
- numpy
- matplotlib
- multiprocessing

### Early Experiments
- gymnasium (Q-Learning)
- numpy
- matplotlib
- tqdm

## References

- **V-Learning**: Rex-L's value-based learning approach for Tetris
- **CEM**: Original Cross-Entropy Method paper with strict parameter adherence
- **Dellacherie Features**: Standard 6-feature evaluation system for Tetris

## License

MIT License 