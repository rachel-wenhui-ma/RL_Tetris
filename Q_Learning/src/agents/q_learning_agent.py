"""
Q-Learning Agent for Tetris
Implements tabular Q-learning algorithm
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, List
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for Tetris
    Uses tabular Q-learning with epsilon-greedy exploration
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent
        
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
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Chosen action (rotation, move)
        """
        if not valid_actions:
            return (0, 5)  # Default action
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            return self.get_best_action(state, valid_actions)
    
    def get_best_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Get the best action according to Q-values
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Best action (rotation, move)
        """
        if not valid_actions:
            return (0, 5)
        
        # Get Q-values for all valid actions
        q_values = []
        for action in valid_actions:
            q_value = self.q_table[self._state_to_key(state)][action]
            q_values.append(q_value)
        
        # Find action with maximum Q-value
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        
        # If multiple actions have the same Q-value, choose randomly
        return random.choice(best_actions)
    
    def update(self, state: Dict[str, Any], action: Tuple[int, int], 
               reward: float, next_state: Dict[str, Any], 
               next_valid_actions: List[Tuple[int, int]], done: bool) -> None:
        """
        Update Q-values using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: Valid actions in next state
            done: Whether episode is done
        """
        current_q = self.q_table[self._state_to_key(state)][action]
        
        if done:
            # Terminal state: no future rewards
            next_max_q = 0
        else:
            # Get maximum Q-value for next state
            next_q_values = []
            for next_action in next_valid_actions:
                next_q = self.q_table[self._state_to_key(next_state)][next_action]
                next_q_values.append(next_q)
            
            next_max_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self._state_to_key(state)][action] = new_q
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert state dictionary to string key for Q-table
        
        Args:
            state: Game state dictionary
            
        Returns:
            String representation of state
        """
        # Extract key features for state representation
        board = state['board']
        current_piece = state['current_piece']
        next_piece = state['next_piece']
        
        # Create simplified state representation
        # Use board height profile and piece information
        height_profile = self._get_height_profile(board)
        holes = self._count_holes(board)
        
        # Create state key
        state_key = f"h{height_profile}_p{current_piece}_{next_piece}_holes{holes}"
        return state_key
    
    def _get_height_profile(self, board: np.ndarray) -> Tuple[int, ...]:
        """
        Get height profile of the board
        
        Args:
            board: Game board array
            
        Returns:
            Tuple of column heights
        """
        heights = []
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    heights.append(board.shape[0] - row)
                    break
            else:
                heights.append(0)
        
        # Discretize heights to reduce state space
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
        """
        Count holes in the board
        
        Args:
            board: Game board array
            
        Returns:
            Number of holes
        """
        holes = 0
        for col in range(board.shape[1]):
            found_block = False
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        
        # Discretize hole count
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
        """
        Update training statistics
        
        Args:
            episode_reward: Total reward for the episode
            episode_steps: Number of steps in the episode
            episode_lines: Number of lines cleared in the episode
        """
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += episode_reward
        self.training_stats['total_steps'] += episode_steps
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_steps)
        self.training_stats['episode_lines'].append(episode_lines)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary of training statistics
        """
        stats = self.training_stats.copy()
        
        if stats['episodes'] > 0:
            stats['avg_reward'] = stats['total_reward'] / stats['episodes']
            stats['avg_steps'] = stats['total_steps'] / stats['episodes']
            stats['avg_lines'] = np.mean(stats['episode_lines'])
        
        return stats
    
    def save_q_table(self, filepath: str) -> None:
        """
        Save Q-table to file
        
        Args:
            filepath: Path to save Q-table
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, filepath: str) -> None:
        """
        Load Q-table from file
        
        Args:
            filepath: Path to load Q-table from
        """
        import pickle
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in q_table_dict.items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value
    
    def get_q_table_size(self) -> int:
        """
        Get the number of states in Q-table
        
        Returns:
            Number of unique states
        """
        return len(self.q_table)
    
    def get_total_q_values(self) -> int:
        """
        Get the total number of Q-values stored
        
        Returns:
            Total number of state-action pairs
        """
        total = 0
        for state_actions in self.q_table.values():
            total += len(state_actions)
        return total 