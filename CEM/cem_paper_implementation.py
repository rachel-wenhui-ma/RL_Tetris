#!/usr/bin/env python3
"""
Paper-Accurate Cross-Entropy Method Implementation for Tetris
Strictly follows the mathematical principles from the original CEM paper
"""

import numpy as np
import random
import gc
import time
import multiprocessing as mp
from datetime import datetime
import json
import os
import logging
from typing import List, Tuple, Dict
from functools import partial

# 导入我们的模块
import sys
sys.path.append('tetris_paper_reproduction/src')
from tetris_env import TetrisEnv
from features import FeatureExtractor, ActionEvaluator

class PaperAccurateCEM:
    """
    Strictly follows the mathematical principles from the original CEM paper
    """
    
    def __init__(self, feature_set="dellacherie", n_processes=8):
        self.feature_set = feature_set
        self.n_processes = n_processes
        
        # Paper parameters
        self.n = 100  # Population size (Paper: n = 100)
        self.rho = 0.1  # Elite ratio (Paper: ρ = 0.1)
        self.elite_size = int(self.n * self.rho)  # Elite size = 10
        
        # Initial distribution parameters for 6 features (Paper: N(0, 100))
        self.n_features = 6
        self.mu = np.zeros(self.n_features)  # Initial mean
        self.sigma = np.sqrt(100) * np.ones(self.n_features)  # Initial std
        
        # Noise strategy (Paper: Zt = max(5 - t/10, 0))
        self.iteration = 0
        
        # Training history
        self.history = {
            'generations': [],
            'best_scores': [],
            'mean_scores': [],
            'std_scores': [],
            'mu_history': [],
            'sigma_history': [],
            'noise_levels': []
        }
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
    def sample_weights(self) -> np.ndarray:
        """
        Sample weight vectors from current distribution N(μt, σt²)
        Paper: Sample n vectors from ft ~ N(μt, σt²)
        """
        weights = np.random.normal(self.mu, self.sigma, (self.n, self.n_features))
        return weights
    
    def evaluate_population(self, weights_population: np.ndarray, games_per_eval: int = 30) -> Tuple[np.ndarray, List[float]]:
        """
        Evaluate performance of each individual in population
        Paper: Each weight vector plays 1 game, evaluate S(w1), ..., S(wn)
        """
        # Parallel evaluation
        with mp.Pool(processes=self.n_processes) as pool:
            eval_func = partial(self._evaluate_single_weights, games_per_eval=games_per_eval)
            results = pool.map(eval_func, weights_population)
        
        scores = np.array([result['score'] for result in results])
        game_times = [result['game_time'] for result in results]
        
        return scores, game_times
    
    def _evaluate_single_weights(self, weights: np.ndarray, games_per_eval: int = 30) -> Dict:
        """
        Evaluate single weight vector (Paper: play 1 game)
        """
        try:
            feature_extractor = FeatureExtractor(self.feature_set)
            action_evaluator = ActionEvaluator(feature_extractor)
            
            scores = []
            total_time = 0
            
            for _ in range(games_per_eval):
                env = TetrisEnv()
                state = env.reset()
                game_score = 0
                move_count = 0
                start_time = time.time()
                
                while not env.game_over and move_count < 1000:
                    try:
                        action = action_evaluator.get_best_action(env, weights)
                        if action is None:
                            break
                        
                        state, reward, done = env.step(action)
                        game_score = env.lines_cleared
                        move_count += 1
                        
                        if done:
                            break
                    except:
                        break
                
                game_time = time.time() - start_time
                scores.append(game_score)
                total_time += game_time
                
                del env
            
            del feature_extractor, action_evaluator
            gc.collect()
            
            return {
                'score': np.mean(scores),
                'game_time': total_time / games_per_eval
            }
            
        except Exception as e:
            return {'score': 0, 'game_time': 0}
    
    def select_elite(self, weights_population: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select elite individuals
        Paper: Select top [ρ·n] best samples, set γt = S(w[ρ·n])
        """
        # Sort by score, select top elite_size
        elite_indices = np.argsort(scores)[-self.elite_size:]
        elite_weights = weights_population[elite_indices]
        elite_scores = scores[elite_indices]
        
        return elite_weights, elite_scores
    
    def update_distribution(self, elite_weights: np.ndarray) -> None:
        """
        Update distribution parameters
        Paper: Equations 2.3 and 2.5
        """
        # Equation 2.3: Mean update
        self.mu = np.mean(elite_weights, axis=0)
        
        # Equation 2.5: Variance update + noise
        # σ²_{t+1} := (Σ_{i∈I} (w_i - μ_{t+1})ᵀ (w_i - μ_{t+1})) / |I| + Z_{t+1}
        
        # Calculate variance
        diff = elite_weights - self.mu
        variance = np.mean(diff**2, axis=0)
        
        # Calculate noise Z_{t+1} = max(5 - t/10, 0)
        noise = np.maximum(5 - self.iteration / 10, 0)
        Z_t = noise * np.ones(self.n_features)
        
        # Update standard deviation
        self.sigma = np.sqrt(variance + Z_t)
        
        # Ensure standard deviation is not too small
        self.sigma = np.maximum(self.sigma, 0.1)
    
    def train(self, max_generations: int = 100, games_per_eval: int = 30, 
              convergence_threshold: float = 1000.0, patience: int = 20) -> Dict:
        """
        Main training loop
        Paper: Iteratively update distribution parameters until reaching sufficiently large γk
        Stop when: 1) Reached max_generations, 2) Performance converged, or 3) No improvement for patience generations
        """
        self.logger.info(f"Starting Paper-Accurate CEM training")
        self.logger.info(f"Parameters: n={self.n}, ρ={self.rho}, elite_size={self.elite_size}")
        self.logger.info(f"Max generations: {max_generations}, Games per eval: {games_per_eval}")
        self.logger.info(f"Convergence threshold: {convergence_threshold} lines, Patience: {patience} generations")
        
        start_time = time.time()
        best_score_ever = 0
        generations_without_improvement = 0
        
        for generation in range(max_generations):
            self.iteration = generation
            gen_start_time = time.time()
            
            # 1. Sample from current distribution
            weights_population = self.sample_weights()
            
            # 2. Evaluate population
            scores, game_times = self.evaluate_population(weights_population, games_per_eval)
            
            # 3. Select elite
            elite_weights, elite_scores = self.select_elite(weights_population, scores)
            
            # 4. Update distribution
            self.update_distribution(elite_weights)
            
            # 5. Evaluate current mean performance (Paper: play 30 games with μt)
            current_performance = self._evaluate_single_weights(self.mu, games_per_eval)
            
            # 6. Record history
            self._record_history(generation, scores, current_performance)
            
            # 7. Output progress
            gen_time = time.time() - gen_start_time
            self._log_progress(generation, scores, current_performance, gen_time)
            
            # 8. Check convergence conditions
            current_best = np.max(scores)
            if current_best > best_score_ever:
                best_score_ever = current_best
                generations_without_improvement = 0
                self.logger.info(f"New best score: {best_score_ever:.1f} lines!")
            else:
                generations_without_improvement += 1
            
            # Check if we should stop
            stop_reason = None
            if current_best >= convergence_threshold:
                stop_reason = f"Reached convergence threshold ({convergence_threshold} lines)"
            elif generations_without_improvement >= patience:
                stop_reason = f"No improvement for {patience} generations"
            elif generation == max_generations - 1:
                stop_reason = f"Reached maximum generations ({max_generations})"
            
            # 9. Save checkpoint
            if (generation + 1) % 10 == 0:
                self._save_checkpoint(generation)
            
            # 10. Stop if convergence conditions met
            if stop_reason:
                self.logger.info(f"Training stopped: {stop_reason}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"Final best score: {best_score_ever:.1f} lines")
        

        
        return self._get_final_results()
    
    def _record_history(self, generation: int, scores: np.ndarray, current_performance: Dict):
        """Record training history"""
        self.history['generations'].append(generation)
        self.history['best_scores'].append(np.max(scores))
        self.history['mean_scores'].append(np.mean(scores))
        self.history['std_scores'].append(np.std(scores))
        self.history['mu_history'].append(self.mu.copy())
        self.history['sigma_history'].append(self.sigma.copy())
        self.history['noise_levels'].append(np.maximum(5 - generation / 10, 0))
    
    def _log_progress(self, generation: int, scores: np.ndarray, current_performance: Dict, gen_time: float):
        """Output training progress"""
        best_score = np.max(scores)
        mean_score = np.mean(scores)
        current_score = current_performance['score']
        noise_level = np.maximum(5 - generation / 10, 0)
        
        # Log basic progress
        self.logger.info(f"Generation {generation+1}: Best={best_score:.1f}, Mean={mean_score:.1f}, "
                        f"Current={current_score:.1f}, Noise={noise_level:.2f}, Time={gen_time:.1f}s")
    
    def _save_checkpoint(self, generation: int):
        """Save checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"cem_paper_checkpoint_gen{generation+1}_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save current state
        checkpoint_data = {
            'generation': generation + 1,
            'mu': self.mu.tolist(),
            'sigma': self.sigma.tolist(),
            'history': self.history,
            'timestamp': timestamp
        }
        
        with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _get_final_results(self) -> Dict:
        """Get final results"""
        return {
            'final_mu': self.mu.tolist(),
            'final_sigma': self.sigma.tolist(),
            'history': self.history,
            'best_score': np.max(self.history['best_scores']),
            'final_mean_score': self.history['mean_scores'][-1] if self.history['mean_scores'] else 0
        }

def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = PaperAccurateCEM(feature_set="dellacherie", n_processes=8)
    
    # Start training
    results = trainer.train(max_generations=100, games_per_eval=30, 
                           convergence_threshold=1000.0, patience=20)
    
    # Output final results
    print(f"\n=== Final Results ===")
    print(f"Best Score: {results['best_score']:.1f} lines")
    print(f"Final Mean Score: {results['final_mean_score']:.1f} lines")
    print(f"Final Weights: {results['final_mu']}")
    print(f"Final Sigma: {results['final_sigma']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"cem_paper_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
