"""
Tetris Reinforcement Learning Environment
Implements the Gymnasium interface for RL training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from .tetris_game import TetrisGame, TetrominoType


class TetrisEnv(gym.Env):
    """
    Tetris environment for reinforcement learning
    Follows the Gymnasium interface
    """
    
    def __init__(self, board_width: int = 10, board_height: int = 20, 
                 reward_type: str = 'standard'):
        """
        Initialize Tetris environment
        
        Args:
            board_width: Width of the game board
            board_height: Height of the game board
            reward_type: Type of reward function ('standard', 'dense', 'sparse')
        """
        super().__init__()
        
        self.board_width = board_width
        self.board_height = board_height
        self.reward_type = reward_type
        
        # Initialize game
        self.game = TetrisGame(board_width, board_height)
        
        # Define action space
        # Actions: (rotation, horizontal_move)
        # rotation: 0-3 (0°, 90°, 180°, 270°)
        # horizontal_move: -5 to +5 (left/right movement)
        self.action_space = spaces.MultiDiscrete([4, 11])  # [rotation, move]
        
        # Define observation space
        # Board state + current piece + next piece
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=2, shape=(board_height, board_width), dtype=np.int8),
            'current_piece': spaces.Discrete(7),  # 7 tetromino types
            'next_piece': spaces.Discrete(7),
            'current_x': spaces.Box(low=-2, high=board_width+2, shape=(), dtype=np.int8),
            'current_y': spaces.Box(low=0, high=board_height, shape=(), dtype=np.int8),
            'current_rotation': spaces.Discrete(4)
        })
        
        # Game statistics
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_lines = 0
        
        # Reward weights
        self.reward_weights = {
            'line_clear': 100,
            'height_penalty': -1,  # 从-2改为-1
            'holes_penalty': -5,   # 从-10改为-5
            'roughness_penalty': -2,  # 从-5改为-2
            'survival': 1,
            'tetris_bonus': 50,  # Bonus for clearing 4 lines at once
            'combo_bonus': 10    # Bonus for consecutive line clears
        }
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset game
        self.game.reset()
        
        # Reset episode statistics
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_lines = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Tuple of (rotation, horizontal_move)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        rotation, horizontal_move = action
        
        # Convert horizontal_move from [0, 10] to [-5, 5]
        horizontal_move = horizontal_move - 5
        
        # Store previous state for reward calculation
        prev_state = {
            'height': self.game.get_height(),
            'holes': self.game.get_holes(),
            'roughness': self.game.get_roughness(),
            'lines_cleared': self.game.lines_cleared
        }
        
        # Apply action
        success = self._apply_action(rotation, horizontal_move)
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, success)
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_steps += 1
        self.episode_lines = self.game.lines_cleared
        
        # Check if episode is done
        terminated = self.game.game_over
        truncated = self.episode_steps >= 1000  # Max steps per episode
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, rotation: int, horizontal_move: int) -> bool:
        """
        Apply the given action to the game
        
        Args:
            rotation: Rotation (0-3)
            horizontal_move: Horizontal movement (-5 to 5)
            
        Returns:
            True if action was successful, False otherwise
        """
        # Set rotation
        for _ in range(rotation):
            if not self.game.rotate_piece():
                return False
        
        # Set horizontal position
        target_x = self.game.board_width // 2 - 2 + horizontal_move
        target_x = max(-2, min(self.game.board_width + 2, target_x))
        
        # Move to target position
        while self.game.current_x < target_x:
            if not self.game.move_piece(1, 0):
                break
        while self.game.current_x > target_x:
            if not self.game.move_piece(-1, 0):
                break
        
        # Drop the piece
        lines_cleared = self.game.drop_piece()
        
        return True
    
    def _calculate_reward(self, prev_state: Dict[str, Any], action_success: bool) -> float:
        """
        Calculate reward for the current step
        
        Args:
            prev_state: Previous game state
            action_success: Whether the action was successful
            
        Returns:
            Reward value
        """
        if not action_success:
            return -100  # Penalty for invalid action
        
        reward = 0
        
        # Line clearing reward
        lines_cleared = self.game.lines_cleared - prev_state['lines_cleared']
        if lines_cleared > 0:
            reward += lines_cleared * self.reward_weights['line_clear']
            
            # Tetris bonus (4 lines at once)
            if lines_cleared == 4:
                reward += self.reward_weights['tetris_bonus']
        
        # Height penalty
        current_height = self.game.get_height()
        height_diff = current_height - prev_state['height']
        reward += height_diff * self.reward_weights['height_penalty']
        
        # Holes penalty
        current_holes = self.game.get_holes()
        holes_diff = current_holes - prev_state['holes']
        reward += holes_diff * self.reward_weights['holes_penalty']
        
        # Roughness penalty
        current_roughness = self.game.get_roughness()
        roughness_diff = current_roughness - prev_state['roughness']
        reward += roughness_diff * self.reward_weights['roughness_penalty']
        
        # Survival reward
        reward += self.reward_weights['survival']
        
        # Game over penalty
        if self.game.game_over:
            reward -= 1000
        
        return reward
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation
        
        Returns:
            Observation dictionary
        """
        game_state = self.game.get_game_state()
        
        return {
            'board': game_state['board_with_piece'].astype(np.int8),
            'current_piece': game_state['current_piece'].value if game_state['current_piece'] else 0,
            'next_piece': game_state['next_piece'].value if game_state['next_piece'] else 0,
            'current_x': np.int8(game_state['current_x']),
            'current_y': np.int8(game_state['current_y']),
            'current_rotation': game_state['current_rotation']
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information
        
        Returns:
            Info dictionary
        """
        return {
            'score': self.game.score,
            'lines_cleared': self.game.lines_cleared,
            'level': self.game.level,
            'height': self.game.get_height(),
            'holes': self.game.get_holes(),
            'roughness': self.game.get_roughness(),
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
            'episode_lines': self.episode_lines
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment
        
        Args:
            mode: Rendering mode ('human', 'rgb_array')
            
        Returns:
            Rendered image (for 'rgb_array' mode)
        """
        if mode == 'rgb_array':
            # Return board as RGB array
            board = self.game.get_board_state()
            rgb_board = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
            
            # Color mapping: 0=black, 1=gray, 2=blue
            rgb_board[board == 0] = [0, 0, 0]      # Empty
            rgb_board[board == 1] = [128, 128, 128] # Placed pieces
            rgb_board[board == 2] = [0, 0, 255]     # Current piece
            
            return rgb_board
        
        elif mode == 'human':
            # Print ASCII representation
            board = self.game.get_board_state()
            print("=" * (self.board_width + 2))
            for row in board:
                print("|", end="")
                for cell in row:
                    if cell == 0:
                        print(" ", end="")
                    elif cell == 1:
                        print("#", end="")
                    else:
                        print("O", end="")
                print("|")
            print("=" * (self.board_width + 2))
            print(f"Score: {self.game.score}, Lines: {self.game.lines_cleared}")
        
        return None
    
    def close(self) -> None:
        """Close the environment"""
        pass
    
    def get_valid_actions(self) -> list:
        """
        Get list of valid actions for current state
        
        Returns:
            List of valid (rotation, move) tuples
        """
        valid_actions = []
        
        for rotation in range(4):
            for move in range(11):  # 0-10
                horizontal_move = move - 5  # Convert to -5 to 5
                
                # Test if this action is valid
                if self._is_valid_action(rotation, horizontal_move):
                    valid_actions.append((rotation, move))
        
        return valid_actions
    
    def _is_valid_action(self, rotation: int, horizontal_move: int) -> bool:
        """
        Check if an action is valid
        
        Args:
            rotation: Rotation (0-3)
            horizontal_move: Horizontal movement (-5 to 5)
            
        Returns:
            True if action is valid, False otherwise
        """
        # Create a copy of the game to test the action
        test_game = TetrisGame(self.board_width, self.board_height)
        test_game.board = self.game.board.copy()
        test_game.current_piece = self.game.current_piece
        test_game.current_x = self.game.current_x
        test_game.current_y = self.game.current_y
        test_game.current_rotation = self.game.current_rotation
        
        # Apply rotation
        for _ in range(rotation):
            if not test_game.rotate_piece():
                return False
        
        # Apply horizontal movement
        target_x = test_game.board_width // 2 - 2 + horizontal_move
        target_x = max(-2, min(test_game.board_width + 2, target_x))
        
        # Check if we can move to target position
        while test_game.current_x < target_x:
            if not test_game.move_piece(1, 0):
                return False
        while test_game.current_x > target_x:
            if not test_game.move_piece(-1, 0):
                return False
        
        return True 