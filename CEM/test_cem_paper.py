#!/usr/bin/env python3
"""
Test script for Paper-Accurate CEM Implementation
"""

import logging
from cem_paper_implementation import PaperAccurateCEM

def test_basic_functionality():
    """Test basic functionality with small parameters"""
    print("Testing Paper-Accurate CEM Implementation...")
    
    # Setup logging to both console and file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_cem_paper_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    print(f"Log file: {log_filename}")
    
    # Create trainer with small parameters for testing
    trainer = PaperAccurateCEM(feature_set="dellacherie", n_processes=4)
    
    print(f"Initial parameters:")
    print(f"  Population size: {trainer.n}")
    print(f"  Elite ratio: {trainer.rho}")
    print(f"  Elite size: {trainer.elite_size}")
    print(f"  Initial mu: {trainer.mu}")
    print(f"  Initial sigma: {trainer.sigma}")
    
    # Test sampling
    print(f"\nTesting weight sampling...")
    weights = trainer.sample_weights()
    print(f"  Sampled weights shape: {weights.shape}")
    print(f"  First weight vector: {weights[0]}")
    
    # Test with very small training (just 2 generations)
    print(f"\nTesting training with 2 generations...")
    try:
        results = trainer.train(
            max_generations=2,
            games_per_eval=2,  # Very small for testing
            convergence_threshold=10000.0,  # High threshold so it won't stop early
            patience=100
        )
        
        print(f"Training completed successfully!")
        print(f"  Best score: {results['best_score']:.1f} lines")
        print(f"  Final weights: {results['final_mu']}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
