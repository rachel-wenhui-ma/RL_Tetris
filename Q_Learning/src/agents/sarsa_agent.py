"""
SARSA Agent for Tetris
Implements SARSA algorithm (placeholder for future implementation)
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, List
from collections import defaultdict


class SarsaAgent:
    """
    SARSA agent for Tetris
    Placeholder implementation - to be completed later
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01):
        """
        Initialize SARSA agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'total_steps': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_lines': []
        }
    
    def get_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Placeholder - same as Q-Learning for now"""
        if not valid_actions:
            return (0, 5)
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            return self.get_best_action(state, valid_actions)
    
    def get_best_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Placeholder - same as Q-Learning for now"""
        if not valid_actions:
            return (0, 5)
        
        q_values = []
        for action in valid_actions:
            q_value = self.q_table[self._state_to_key(state)][action]
            q_values.append(q_value)
        
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state: Dict[str, Any], action: Tuple[int, int], 
               reward: float, next_state: Dict[str, Any], 
               next_action: Tuple[int, int], done: bool) -> None:
        """Placeholder - same as Q-Learning for now"""
        current_q = self.q_table[self._state_to_key(state)][action]
        
        if done:
            next_q = 0
        else:
            next_q = self.q_table[self._state_to_key(next_state)][next_action]
        
        # SARSA update rule (same as Q-Learning for now)
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[self._state_to_key(state)][action] = new_q
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Placeholder - same as Q-Learning for now"""
        board = state['board']
        current_piece = state['current_piece']
        next_piece = state['next_piece']
        
        height_profile = self._get_height_profile(board)
        holes = self._count_holes(board)
        
        state_key = f"h{height_profile}_p{current_piece}_{next_piece}_holes{holes}"
        return state_key
    
    def _get_height_profile(self, board: np.ndarray) -> Tuple[int, ...]:
        """Placeholder - same as Q-Learning for now"""
        heights = []
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    heights.append(board.shape[0] - row)
                    break
            else:
                heights.append(0)
        
        discrete_heights = []
        for height in heights:
            if height == 0:
                discrete_heights.append(0)
            elif height <= 5:
                discrete_heights.append(1)
            elif height <= 10:
                discrete_heights.append(2)
            elif height <= 15:
                discrete_heights.append(3)
            else:
                discrete_heights.append(4)
        
        return tuple(discrete_heights)
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Placeholder - same as Q-Learning for now"""
        holes = 0
        for col in range(board.shape[1]):
            found_block = False
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        
        if holes == 0:
            return 0
        elif holes <= 3:
            return 1
        elif holes <= 6:
            return 2
        else:
            return 3
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_training_stats(self, episode_reward: float, episode_steps: int, 
                             episode_lines: int) -> None:
        """Update training statistics"""
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += episode_reward
        self.training_stats['total_steps'] += episode_steps
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_steps)
        self.training_stats['episode_lines'].append(episode_lines)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = self.training_stats.copy()
        
        if stats['episodes'] > 0:
            stats['avg_reward'] = stats['total_reward'] / stats['episodes']
            stats['avg_steps'] = stats['total_steps'] / stats['episodes']
            stats['avg_lines'] = np.mean(stats['episode_lines'])
        
        return stats
    
    def get_q_table_size(self) -> int:
        """Get the number of states in Q-table"""
        return len(self.q_table)
    
    def get_total_q_values(self) -> int:
        """Get the total number of Q-values stored"""
        total = 0
        for state_actions in self.q_table.values():
            total += len(state_actions)
        return total 