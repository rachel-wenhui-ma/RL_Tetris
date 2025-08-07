"""
Feature extraction for Tetris RL algorithms
Implements features from various research papers
"""

import numpy as np
from typing import List, Tuple, Dict

class FeatureExtractor:
    """Extract features from Tetris state for evaluation"""
    
    def __init__(self, feature_set: str = "dellacherie"):
        """
        Initialize feature extractor
        
        Args:
            feature_set: Which feature set to use
                - "dellacherie": 6 features from Dellacherie (2003)
                - "bertsekas": 21 features from Bertsekas & Tsitsiklis (1996)
                - "lagoudakis": 9 features from Lagoudakis et al. (2002)
        """
        self.feature_set = feature_set
    
    def extract_features(self, env) -> np.ndarray:
        """Extract features from current environment state"""
        if self.feature_set == "dellacherie":
            return self._extract_dellacherie_features(env)
        elif self.feature_set == "bertsekas":
            return self._extract_bertsekas_features(env)
        elif self.feature_set == "lagoudakis":
            return self._extract_lagoudakis_features(env)
        else:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
    
    def _extract_dellacherie_features(self, env) -> np.ndarray:
        """
        Extract Dellacherie features (6 features)
        From: Dellacherie (2003) - Hand-crafted Tetris AI
        """
        features = []
        
        # 1. Number of holes
        holes = env.get_holes()
        features.append(holes)
        
        # 2. Landing height of the piece
        # For current piece placement evaluation
        landing_height = self._get_landing_height(env)
        features.append(landing_height)
        
        # 3. Number of row transitions
        row_transitions = self._count_row_transitions(env)
        features.append(row_transitions)
        
        # 4. Number of column transitions
        col_transitions = self._count_column_transitions(env)
        features.append(col_transitions)
        
        # 5. Cumulative wells
        cumulative_wells = self._get_cumulative_wells(env)
        features.append(cumulative_wells)
        
        # 6. Eroded cells
        eroded_cells = self._get_eroded_cells(env)
        features.append(eroded_cells)
        
        return np.array(features, dtype=float)
    
    def _extract_bertsekas_features(self, env) -> np.ndarray:
        """
        Extract Bertsekas features (21 features)
        From: Bertsekas & Tsitsiklis (1996)
        """
        features = []
        
        # Basic features
        holes = env.get_holes()
        features.append(holes)
        
        # Height of highest column (pile height)
        heights = env.get_height_profile()
        max_height = np.max(heights)
        features.append(max_height)
        
        # Height of each column (10 features)
        for height in heights:
            features.append(height)
        
        # Difference in height between consecutive columns (9 features)
        for i in range(len(heights) - 1):
            diff = heights[i+1] - heights[i]
            features.append(diff)
        
        return np.array(features, dtype=float)
    
    def _extract_lagoudakis_features(self, env) -> np.ndarray:
        """
        Extract Lagoudakis features (9 features)
        From: Lagoudakis et al. (2002)
        """
        features = []
        
        # Basic features
        holes = env.get_holes()
        features.append(holes)
        
        heights = env.get_height_profile()
        max_height = np.max(heights)
        features.append(max_height)
        
        # Sum of differences in height of consecutive columns
        height_diffs = np.diff(heights)
        sum_diffs = np.sum(np.abs(height_diffs))
        features.append(sum_diffs)
        
        # Mean height
        mean_height = np.mean(heights)
        features.append(mean_height)
        
        # Change in features between current and next state
        # (This would require simulating next state, simplified here)
        features.extend([0, 0, 0, 0, 0])  # Placeholder for state changes
        
        return np.array(features, dtype=float)
    
    def _get_landing_height(self, env) -> float:
        """Get landing height of current piece"""
        if env.current_piece is None:
            return 0.0
        
        # Find the lowest valid position for current piece
        min_y = env.height
        for action in env.get_valid_actions():
            x_pos, rotation = action
            # Simulate dropping piece
            temp_y = 0
            while not env._check_collision(dx=x_pos-env.current_x, dy=temp_y+1, rotation=rotation):
                temp_y += 1
            landing_y = temp_y
            min_y = min(min_y, landing_y)
        
        return float(min_y)
    
    def _count_row_transitions(self, env) -> int:
        """Count row transitions (empty to filled or vice versa)"""
        transitions = 0
        for y in range(env.height):
            for x in range(env.width - 1):
                current = env.grid[y, x]
                next_cell = env.grid[y, x + 1]
                if current != next_cell:
                    transitions += 1
        return transitions
    
    def _count_column_transitions(self, env) -> int:
        """Count column transitions (empty to filled or vice versa)"""
        transitions = 0
        for x in range(env.width):
            for y in range(env.height - 1):
                current = env.grid[y, x]
                next_cell = env.grid[y + 1, x]
                if current != next_cell:
                    transitions += 1
        return transitions
    
    def _get_cumulative_wells(self, env) -> float:
        """Calculate cumulative wells score"""
        wells = env.get_wells()
        cumulative = 0
        for well_depth in wells:
            # Sum of depths: 1 + 2 + 3 + ... + depth
            cumulative += (well_depth * (well_depth + 1)) // 2
        return float(cumulative)
    
    def _get_eroded_cells(self, env) -> float:
        """Calculate eroded cells (holes in cleared lines)"""
        # This is a simplified version
        # In practice, this would track holes in lines that get cleared
        holes = env.get_holes()
        lines_cleared = env.lines_cleared
        return float(holes * lines_cleared)
    
    def evaluate_state(self, env, weights: np.ndarray) -> float:
        """
        Evaluate a state using linear combination of features
        
        Args:
            env: Tetris environment
            weights: Feature weights
            
        Returns:
            Evaluation score
        """
        features = self.extract_features(env)
        return np.dot(features, weights)
    
    def get_feature_names(self) -> List[str]:
        """Get names of features"""
        if self.feature_set == "dellacherie":
            return [
                "holes",
                "landing_height", 
                "row_transitions",
                "column_transitions",
                "cumulative_wells",
                "eroded_cells"
            ]
        elif self.feature_set == "bertsekas":
            names = ["holes", "max_height"]
            names.extend([f"height_{i}" for i in range(10)])
            names.extend([f"height_diff_{i}" for i in range(9)])
            return names
        elif self.feature_set == "lagoudakis":
            return [
                "holes",
                "max_height",
                "sum_height_diffs", 
                "mean_height",
                "holes_change",
                "max_height_change",
                "sum_diffs_change",
                "mean_height_change",
                "lines_cleared"
            ]
        else:
            return []


class ActionEvaluator:
    """Evaluate actions using feature-based evaluation"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def evaluate_actions(self, env, weights: np.ndarray) -> List[Tuple[Tuple[int, int], float]]:
        """
        Evaluate all valid actions for current state
        
        Args:
            env: Tetris environment
            weights: Feature weights
            
        Returns:
            List of (action, score) tuples
        """
        valid_actions = env.get_valid_actions()
        action_scores = []
        
        for action in valid_actions:
            # Create a copy of environment to simulate action
            temp_env = self._copy_env(env)
            
            # Take action
            temp_env.step(action)
            
            # Evaluate resulting state
            score = self.feature_extractor.evaluate_state(temp_env, weights)
            action_scores.append((action, score))
        
        return action_scores
    
    def get_best_action(self, env, weights: np.ndarray) -> Tuple[int, int]:
        """Get the best action according to current weights"""
        action_scores = self.evaluate_actions(env, weights)
        if not action_scores:
            return None
        best_action = max(action_scores, key=lambda x: x[1])[0]
        return best_action
    
    def _copy_env(self, env):
        """Create a copy of environment for simulation"""
        # Create a new environment instance instead of deep copy
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