# Paper-Accurate Cross-Entropy Method for Tetris

This directory contains a strictly paper-accurate implementation of the Cross-Entropy Method (CEM) for Tetris, following the mathematical principles from the original CEM research paper.

## Implementation Details

### Core Algorithm
- **Population size**: 100 (Paper standard: n = 100)
- **Elite ratio**: 0.1 (Paper standard: ρ = 0.1)
- **Elite size**: 10 (Paper standard: elite_size = 10)
- **Initial distribution**: N(0, 100) (Paper standard)
- **Games per evaluation**: 30 (Paper standard)
- **Noise strategy**: Zt = max(5 - t/10, 0) (Paper standard)

### Training Strategy
- **Adaptive stopping**: Convergence threshold, patience, max generations
- **Parallel processing**: Multiprocessing support for efficient evaluation
- **Checkpointing**: Automatic saving of training state and results

## File Structure

```
CEM/
├── cem_paper_implementation.py    # Core CEM implementation class
├── run_cem_paper.py              # Main training script
├── test_cem_paper.py             # Test script for verification
├── requirements.txt               # Python dependencies
├── README_CEM_Paper.md           # This documentation
├── cem_paper_training_results.png # Training visualization
└── training_results/              # All training outputs and logs
    ├── parallel_cem_results_*/    # Individual generation results
    ├── *.log                      # Training logs
    ├── *.json                     # Training state files
    └── *.npy                      # Weight arrays
```

## Core Components

### 1. PaperAccurateCEM Class
- **Location**: `cem_paper_implementation.py`
- **Features**: 
  - Strict paper parameter adherence
  - Parallel population evaluation
  - Adaptive noise strategy
  - Comprehensive logging and checkpointing

### 2. Tetris Environment
- **Location**: `tetris_paper_reproduction/src/`
- **Features**:
  - Dellacherie feature extraction (6 features)
  - Action evaluation with placement simulation
  - Standard Tetris game rules

### 3. Training Results
- **Location**: `training_results/`
- **Contents**:
  - Generation-by-generation results
  - Weight evolution history
  - Performance metrics
  - Training logs

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
python test_cem_paper.py
```

### Training
```bash
python run_cem_paper.py
```

## Training Parameters

The implementation uses the following default parameters:
- **Max generations**: 100
- **Games per evaluation**: 30
- **Parallel processes**: 8
- **Convergence threshold**: 1000.0 lines
- **Patience**: 20 generations

## Performance Results

### Paper-Accurate Implementation (4 generations completed)
- **Generation 1**: Best=304.1, Mean=4.1, Time=511.0s
- **Generation 2**: Best=96.3, Mean=11.6, Time=1262.3s
- **Generation 3**: Best=350.2, Mean=83.0, Time=3108.1s
- **Generation 4**: Best=400.1, Mean=179.5, Time=7452.8s

### Key Features
- **Strict paper adherence**: All parameters match original research
- **Parallel efficiency**: 8-process parallel evaluation
- **Robust logging**: Comprehensive training progress tracking
- **Checkpointing**: Automatic state saving and recovery
