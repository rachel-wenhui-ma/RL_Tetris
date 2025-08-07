"""
Cross-Entropy Method for Tetris
Reproduction of: Szita, I. & Lörincz, A. (2006). Learning Tetris using the noisy cross-entropy method.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tetris_env import TetrisEnv
from features import FeatureExtractor, ActionEvaluator


class CrossEntropyTetris:
    """
    Cross-Entropy Method for Tetris
    Based on Szita & Lörincz (2006)
    """
    
    def __init__(self, 
                 feature_set: str = "dellacherie",
                 population_size: int = 50,
                 elite_size: int = 10,
                 noise_std: float = 1.0,
                 noise_decay: float = 0.99):
        """
        Initialize Cross-Entropy method
        
        Args:
            feature_set: Which feature set to use
            population_size: Number of parameter vectors to sample
            elite_size: Number of best vectors to keep
            noise_std: Standard deviation of noise for exploration
            noise_decay: Decay factor for noise
        """
        self.feature_set = feature_set
        self.population_size = population_size
        self.elite_size = elite_size
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        
        # Initialize feature extractor and action evaluator
        self.feature_extractor = FeatureExtractor(feature_set)
        self.action_evaluator = ActionEvaluator(self.feature_extractor)
        
        # Get number of features
        self.n_features = len(self.feature_extractor.get_feature_names())
        
        # Initialize parameter distribution
        self.mean = np.zeros(self.n_features)
        self.std = np.ones(self.n_features) * noise_std
        
        # Training history
        self.history = {
            'generations': [],
            'best_scores': [],
            'mean_scores': [],
            'std_scores': [],
            'noise_levels': []
        }
    
    def sample_parameters(self) -> np.ndarray:
        """Sample a parameter vector from current distribution"""
        return np.random.normal(self.mean, self.std)
    
    def evaluate_parameters(self, weights: np.ndarray, n_games: int = 10) -> float:
        """
        Evaluate a parameter vector by playing multiple games
        
        Args:
            weights: Feature weights
            n_games: Number of games to play
            
        Returns:
            Average score across games
        """
        scores = []
        
        for _ in range(n_games):
            env = TetrisEnv()
            state = env.reset()
            game_score = 0
            
            while not env.game_over:
                # Get best action according to current weights
                action = self.action_evaluator.get_best_action(env, weights)
                
                # Take action
                state, reward, done = env.step(action)
                game_score = env.lines_cleared  # Use lines cleared as score
                
                if done:
                    break
            
            scores.append(game_score)
        
        return np.mean(scores)
    
    def update_distribution(self, elite_weights: List[np.ndarray]):
        """
        Update parameter distribution based on elite vectors
        
        Args:
            elite_weights: List of best parameter vectors
        """
        if not elite_weights:
            return
        
        # Convert to numpy array
        elite_array = np.array(elite_weights)
        
        # Update mean and std
        self.mean = np.mean(elite_array, axis=0)
        self.std = np.std(elite_array, axis=0)
        
        # Add minimum noise to prevent collapse
        min_std = 0.1
        self.std = np.maximum(self.std, min_std)
        
        # Decay noise
        self.noise_std *= self.noise_decay
    
    def train(self, n_generations: int = 100, games_per_eval: int = 5):
        """
        Train the Cross-Entropy method
        
        Args:
            n_generations: Number of generations to train
            games_per_eval: Number of games per parameter evaluation
        """
        print(f"🎮 Starting Cross-Entropy training for Tetris")
        print(f"   Feature set: {self.feature_set}")
        print(f"   Population size: {self.population_size}")
        print(f"   Elite size: {self.elite_size}")
        print(f"   Generations: {n_generations}")
        print(f"   Games per evaluation: {games_per_eval}")
        print("=" * 60)
        
        best_score_overall = 0
        best_weights_overall = None
        
        for generation in tqdm(range(n_generations), desc="Training"):
            # Sample population
            population = []
            for _ in range(self.population_size):
                weights = self.sample_parameters()
                population.append(weights)
            
            # Evaluate population
            scores = []
            for weights in population:
                score = self.evaluate_parameters(weights, games_per_eval)
                scores.append(score)
            
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]  # Descending order
            elite_indices = sorted_indices[:self.elite_size]
            elite_weights = [population[i] for i in elite_indices]
            elite_scores = [scores[i] for i in elite_indices]
            
            # Update distribution
            self.update_distribution(elite_weights)
            
            # Track best overall
            generation_best_score = max(scores)
            if generation_best_score > best_score_overall:
                best_score_overall = generation_best_score
                best_weights_overall = population[scores.index(generation_best_score)]
            
            # Record history
            self.history['generations'].append(generation)
            self.history['best_scores'].append(generation_best_score)
            self.history['mean_scores'].append(np.mean(scores))
            self.history['std_scores'].append(np.std(scores))
            self.history['noise_levels'].append(self.noise_std)
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best={generation_best_score:.1f}, "
                      f"Mean={np.mean(scores):.1f}, Noise={self.noise_std:.3f}")
        
        print("=" * 60)
        print(f"✅ Training completed!")
        print(f"   Best score: {best_score_overall:.1f} lines")
        print(f"   Best weights: {best_weights_overall}")
        
        return best_weights_overall, self.history
    
    def test_agent(self, weights: np.ndarray, n_games: int = 100) -> Dict:
        """
        Test the trained agent
        
        Args:
            weights: Trained weights
            n_games: Number of games to play
            
        Returns:
            Test results
        """
        print(f"🧪 Testing agent with {n_games} games...")
        
        scores = []
        lines_cleared = []
        pieces_placed = []
        
        for i in tqdm(range(n_games), desc="Testing"):
            env = TetrisEnv()
            state = env.reset()
            
            while not env.game_over:
                action = self.action_evaluator.get_best_action(env, weights)
                state, reward, done = env.step(action)
                
                if done:
                    break
            
            scores.append(env.score)
            lines_cleared.append(env.lines_cleared)
            pieces_placed.append(env.pieces_placed)
        
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_lines': np.mean(lines_cleared),
            'std_lines': np.std(lines_cleared),
            'mean_pieces': np.mean(pieces_placed),
            'std_pieces': np.std(pieces_placed),
            'max_score': np.max(scores),
            'max_lines': np.max(lines_cleared),
            'scores': scores,
            'lines_cleared': lines_cleared,
            'pieces_placed': pieces_placed
        }
        
        print(f"   Mean lines cleared: {results['mean_lines']:.1f} ± {results['std_lines']:.1f}")
        print(f"   Max lines cleared: {results['max_lines']}")
        print(f"   Mean pieces placed: {results['mean_pieces']:.1f} ± {results['std_pieces']:.1f}")
        
        return results
    
    def save_results(self, weights: np.ndarray, history: Dict, test_results: Dict, 
                    save_dir: str = "../results"):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save weights
        weights_file = os.path.join(save_dir, f"cross_entropy_weights_{timestamp}.npy")
        np.save(weights_file, weights)
        
        # Save history
        history_file = os.path.join(save_dir, f"cross_entropy_history_{timestamp}.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save test results
        test_file = os.path.join(save_dir, f"cross_entropy_test_{timestamp}.json")
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"💾 Results saved to {save_dir}")
    
    def plot_training_history(self, history: Dict, save_dir: str = "../results/plots"):
        """Plot training history"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Best scores
        axes[0, 0].plot(history['generations'], history['best_scores'])
        axes[0, 0].set_title('Best Scores')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Lines Cleared')
        
        # Mean scores
        axes[0, 1].plot(history['generations'], history['mean_scores'])
        axes[0, 1].set_title('Mean Scores')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Lines Cleared')
        
        # Standard deviation
        axes[1, 0].plot(history['generations'], history['std_scores'])
        axes[1, 0].set_title('Score Standard Deviation')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Standard Deviation')
        
        # Noise level
        axes[1, 1].plot(history['generations'], history['noise_levels'])
        axes[1, 1].set_title('Noise Level')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Noise Standard Deviation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cross_entropy_training_{timestamp}.png"))
        plt.close()
        
        print(f"📊 Training plots saved to {save_dir}")


def main():
    """Main training script"""
    # Initialize Cross-Entropy method
    ce_agent = CrossEntropyTetris(
        feature_set="dellacherie",
        population_size=30,  # Smaller for faster training
        elite_size=5,
        noise_std=1.0,
        noise_decay=0.99
    )
    
    # Train
    best_weights, history = ce_agent.train(
        n_generations=50,  # Fewer generations for testing
        games_per_eval=3   # Fewer games for faster training
    )
    
    # Test
    test_results = ce_agent.test_agent(best_weights, n_games=20)
    
    # Save results
    ce_agent.save_results(best_weights, history, test_results)
    ce_agent.plot_training_history(history)
    
    print("🎉 Cross-Entropy training completed!")


if __name__ == "__main__":
    main() 