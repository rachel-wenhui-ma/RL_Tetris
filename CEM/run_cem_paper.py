#!/usr/bin/env python3
"""
Run Paper-Accurate CEM Implementation
"""

import logging
import os
from datetime import datetime
from cem_paper_implementation import PaperAccurateCEM

def setup_logging():
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cem_paper_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

def main():
    """Main function"""
    # Setup logging
    log_filename = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting Paper-Accurate CEM Training")
    logger.info("=" * 60)
    
    # Training parameters
    max_generations = 500    # Maximum generations (can stop earlier)
    games_per_eval = 30      # Paper: 30 games per evaluation
    n_processes = 16         # Number of parallel processes (utilize 16 cores)
    convergence_threshold = 1000000.0  # Stop if we reach 1,000,000+ lines
    patience = 50            # Stop if no improvement for 50 generations
    
    logger.info(f"Training Parameters:")
    logger.info(f"  Max generations: {max_generations}")
    logger.info(f"  Games per evaluation: {games_per_eval}")
    logger.info(f"  Parallel processes: {n_processes}")
    logger.info(f"  Population size: 100 (Paper standard)")
    logger.info(f"  Elite ratio: 0.1 (Paper standard)")
    logger.info(f"  Initial distribution: N(0, 100) (Paper standard)")
    logger.info(f"  Noise strategy: Zt = max(5 - t/10, 0) (Paper standard)")
    logger.info(f"  Convergence threshold: {convergence_threshold} lines")
    logger.info(f"  Patience: {patience} generations")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = PaperAccurateCEM(
        feature_set="dellacherie",
        n_processes=n_processes
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        results = trainer.train(
            max_generations=max_generations,
            games_per_eval=games_per_eval,
            convergence_threshold=convergence_threshold,
            patience=patience
        )
        
        # Output final results
        logger.info("=" * 60)
        logger.info("Training completed! Final results:")
        logger.info(f"  Best score: {results['best_score']:.1f} lines")
        logger.info(f"  Final mean score: {results['final_mean_score']:.1f} lines")
        logger.info(f"  Final weights: {results['final_mu']}")
        logger.info(f"  Final sigma: {results['final_sigma']}")
        logger.info("=" * 60)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"cem_paper_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Detailed log saved to: {log_filename}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
