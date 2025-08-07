"""
Random Agent for Tetris
Implements random action selection for baseline comparison
"""

import random
from typing import Dict, Any, Tuple, List


class RandomAgent:
    """
    Random agent for Tetris
    Selects actions randomly for baseline comparison
    """
    
    def __init__(self):
        """Initialize random agent"""
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
        Choose action randomly
        
        Args:
            state: Current game state (ignored)
            valid_actions: List of valid actions
            
        Returns:
            Randomly chosen action
        """
        if not valid_actions:
            return (0, 5)  # Default action
        
        return random.choice(valid_actions)
    
    def update(self, state: Dict[str, Any], action: Tuple[int, int], 
               reward: float, next_state: Dict[str, Any], 
               next_valid_actions: List[Tuple[int, int]], done: bool) -> None:
        """
        Update method (does nothing for random agent)
        
        Args:
            state: Current state (ignored)
            action: Action taken (ignored)
            reward: Reward received (ignored)
            next_state: Next state (ignored)
            next_valid_actions: Next valid actions (ignored)
            done: Whether episode is done (ignored)
        """
        pass  # Random agent doesn't learn
    
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
            stats['avg_lines'] = sum(stats['episode_lines']) / len(stats['episode_lines'])
        
        return stats 