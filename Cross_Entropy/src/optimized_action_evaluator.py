"""
Optimized Action Evaluator for Tetris
Only evaluates final landing positions, not all intermediate states
"""

import numpy as np
from typing import List, Tuple, Optional
from features import FeatureExtractor

class OptimizedActionEvaluator:
    """Optimized action evaluator that only considers final landing positions"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def evaluate_actions(self, env, weights: np.ndarray) -> List[Tuple[Tuple[int, int], float]]:
        """
        Evaluate all final landing positions for current state
        
        Args:
            env: Tetris environment (must have get_final_landing_positions method)
            weights: Feature weights
            
        Returns:
            List of (action, score) tuples
        """
        # Get final landing positions instead of all valid actions
        if hasattr(env, 'get_final_landing_positions'):
            landing_positions = env.get_final_landing_positions()
        else:
            # Fallback to regular valid actions if method doesn't exist
            landing_positions = env.get_valid_actions()
        
        action_scores = []
        
        for action in landing_positions:
            # Create a copy of environment to simulate action
            temp_env = self._copy_env(env)
            
            # Simulate the action (place piece directly at landing position)
            score = self._simulate_landing_action(temp_env, action, weights)
            action_scores.append((action, score))
        
        return action_scores
    
    def get_best_action(self, env, weights: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get the best action according to current weights"""
        action_scores = self.evaluate_actions(env, weights)
        if not action_scores:
            return None
        best_action = max(action_scores, key=lambda x: x[1])[0]
        return best_action
    
    def _simulate_landing_action(self, env, action: Tuple[int, int], weights: np.ndarray) -> float:
        """
        Simulate placing a piece at the final landing position
        More efficient than simulating the full drop process
        """
        x_pos, rotation = action
        
        # Set piece position and rotation
        env.current_x = x_pos
        env.current_rotation = rotation
        
        # Find the landing position (drop to bottom)
        shape = env._get_piece_shape(env.current_piece, rotation)
        piece_height, piece_width = shape.shape
        
        # Find the lowest valid position
        landing_y = 0
        while not env._check_collision(dy=landing_y+1, rotation=rotation):
            landing_y += 1
        
        # Place piece at landing position
        env.current_y = landing_y
        
        # Place the piece on the grid
        for y in range(piece_height):
            for x in range(piece_width):
                if shape[y, x] == 1:
                    grid_y = landing_y + y
                    grid_x = x_pos + x
                    if 0 <= grid_y < env.height and 0 <= grid_x < env.width:
                        env.grid[grid_y, grid_x] = 1
        
        # Clear lines and update score
        env._clear_lines()
        
        # Evaluate the resulting state
        score = self.feature_extractor.evaluate_state(env, weights)
        
        return score
    
    def _copy_env(self, env):
        """Create a copy of environment for simulation"""
        # Support both TetrisEnv and ImprovedTetrisEnv
        if hasattr(env, 'max_steps'):  # ImprovedTetrisEnv
            from tetris_env_improved import ImprovedTetrisEnv
            new_env = ImprovedTetrisEnv(env.height, env.width, env.max_steps)
            new_env.consecutive_no_actions = env.consecutive_no_actions
            new_env.max_consecutive_no_actions = env.max_consecutive_no_actions
            new_env.steps_taken = env.steps_taken
        else:  # TetrisEnv
            from tetris_env import TetrisEnv
            new_env = TetrisEnv(env.height, env.width)
        
        # Copy common attributes
        new_env.grid = env.grid.copy()
        new_env.current_piece = env.current_piece
        new_env.current_x = env.current_x
        new_env.current_y = env.current_y
        new_env.current_rotation = env.current_rotation
        new_env.lines_cleared = env.lines_cleared
        new_env.score = env.score
        new_env.game_over = env.game_over
        new_env.pieces_placed = env.pieces_placed
        new_env.piece_sequence = env.piece_sequence.copy()
        new_env.sequence_index = env.sequence_index
        return new_env 