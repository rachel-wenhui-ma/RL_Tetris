"""
Test Cross-Entropy with optimized action evaluation
"""

import sys
import os
import time
sys.path.append('src')

# Import the improved environment and optimized action evaluator
from src.tetris_env_improved import ImprovedTetrisEnv
from src.features import FeatureExtractor
from src.optimized_action_evaluator import OptimizedActionEvaluator

class OptimizedCrossEntropyTetris:
    """Cross-Entropy method using improved environment and optimized action evaluation"""
    
    def __init__(self, 
                 feature_set: str = "dellacherie",
                 population_size: int = 3,
                 elite_size: int = 1,
                 noise_std: float = 1.0,
                 noise_decay: float = 0.995):
        """
        Initialize Cross-Entropy method with optimized action evaluation
        """
        self.feature_set = feature_set
        self.population_size = population_size
        self.elite_size = elite_size
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        
        # Initialize feature extractor and optimized action evaluator
        self.feature_extractor = FeatureExtractor(feature_set)
        self.action_evaluator = OptimizedActionEvaluator(self.feature_extractor)
        
        # Get number of features
        self.n_features = len(self.feature_extractor.get_feature_names())
        
        # Initialize parameter distribution
        import numpy as np
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
    
    def sample_parameters(self):
        """Sample a parameter vector from current distribution"""
        import numpy as np
        return np.random.normal(self.mean, self.std)
    
    def evaluate_parameters(self, weights, n_games: int = 1) -> float:
        """
        Evaluate a parameter vector using improved environment and optimized action evaluation
        """
        import numpy as np
        scores = []
        
        for _ in range(n_games):
            env = ImprovedTetrisEnv()
            state = env.reset()
            game_score = 0
            
            while not env.game_over:
                # Get best action according to current weights using optimized evaluation
                action = self.action_evaluator.get_best_action(env, weights)
                if action is None:
                    break
                # Take action
                state, reward, done = env.step(action)
                game_score = env.lines_cleared  # Use lines cleared as score
                
                if done:
                    break
            
            scores.append(game_score)
        
        return np.mean(scores)
    
    def update_distribution(self, elite_weights):
        """Update parameter distribution based on elite vectors"""
        import numpy as np
        
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
    
    def train(self, n_generations: int = 5, games_per_eval: int = 2):
        """
        Train the Cross-Entropy method with optimized action evaluation
        """
        import numpy as np
        from tqdm import tqdm
        
        print(f"🎮 Starting Optimized Cross-Entropy training for Tetris")
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
            if generation % 2 == 0:
                print(f"Generation {generation}: Best={generation_best_score:.1f}, "
                      f"Mean={np.mean(scores):.1f}, Noise={self.noise_std:.3f}")
        
        print("=" * 60)
        print(f"✅ Training completed!")
        print(f"   Best score: {best_score_overall:.1f} lines")
        print(f"   Best weights: {best_weights_overall}")
        
        return best_weights_overall, self.history

def test_optimized_ce():
    """Test Cross-Entropy with optimized action evaluation"""
    print("🚀 Testing optimized Cross-Entropy method...")
    
    # Initialize with baseline parameters
    ce_agent = OptimizedCrossEntropyTetris(
        feature_set="dellacherie",
        population_size=3,    # Keep small for testing
        elite_size=1,         # Keep small for testing
        noise_std=1.0,
        noise_decay=0.995     # Slightly slower decay
    )
    
    print(f"Agent initialized with {ce_agent.n_features} features")
    
    # Train with optimized action evaluation
    print("Starting optimized training...")
    start_time = time.time()
    
    best_weights, history = ce_agent.train(
        n_generations=5,      # 5 generations
        games_per_eval=2      # 2 games per evaluation
    )
    
    total_time = time.time() - start_time
    
    print(f"✅ Optimized test completed in {total_time:.2f}s!")
    print(f"Best weights: {best_weights}")
    print(f"Best score: {max(history['best_scores'])}")
    print(f"Score progression: {[f'{s:.1f}' for s in history['best_scores']]}")
    
    return best_weights, history

if __name__ == "__main__":
    test_optimized_ce() 