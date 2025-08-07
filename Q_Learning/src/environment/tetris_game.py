"""
Tetris Game Core Logic
Implements the basic Tetris game mechanics without RL components
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum


class TetrominoType(Enum):
    """Tetromino piece types"""
    I = 0
    O = 1
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6


class TetrisGame:
    """
    Core Tetris game implementation
    Handles game logic, piece movement, line clearing, etc.
    """
    
    def __init__(self, board_width: int = 10, board_height: int = 20):
        """
        Initialize Tetris game
        
        Args:
            board_width: Width of the game board
            board_height: Height of the game board
        """
        self.board_width = board_width
        self.board_height = board_height
        self.board = np.zeros((board_height, board_width), dtype=int)
        
        # Game state
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.current_rotation = 0
        self.next_piece = None
        
        # Game statistics
        self.lines_cleared = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        
        # Tetromino shapes (4x4 matrices)
        self.tetrominoes = {
            TetrominoType.I: np.array([
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.O: np.array([
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.T: np.array([
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.S: np.array([
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.Z: np.array([
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.J: np.array([
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            TetrominoType.L: np.array([
                [0, 0, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ])
        }
        
        # Initialize first piece
        self._spawn_piece()
    
    def _spawn_piece(self) -> None:
        """Spawn a new piece at the top of the board"""
        if self.next_piece is None:
            self.next_piece = random.choice(list(TetrominoType))
        
        self.current_piece = self.next_piece
        self.next_piece = random.choice(list(TetrominoType))
        self.current_rotation = 0
        self.current_x = self.board_width // 2 - 2
        self.current_y = 0
        
        # Check if game is over
        if not self._is_valid_position(self.current_x, self.current_y, self.current_rotation):
            self.game_over = True
    
    def _get_piece_shape(self, piece_type: TetrominoType, rotation: int = 0) -> np.ndarray:
        """Get the shape of a piece with given rotation"""
        shape = self.tetrominoes[piece_type].copy()
        
        # Apply rotation
        for _ in range(rotation):
            shape = np.rot90(shape, k=-1)
        
        return shape
    
    def _is_valid_position(self, x: int, y: int, rotation: int) -> bool:
        """Check if a piece position is valid"""
        shape = self._get_piece_shape(self.current_piece, rotation)
        
        for row in range(4):
            for col in range(4):
                if shape[row, col] == 1:
                    board_x = x + col
                    board_y = y + row
                    
                    # Check boundaries
                    if (board_x < 0 or board_x >= self.board_width or 
                        board_y >= self.board_height):
                        return False
                    
                    # Check collision with existing pieces
                    if board_y >= 0 and self.board[board_y, board_x] == 1:
                        return False
        
        return True
    
    def move_piece(self, dx: int, dy: int) -> bool:
        """
        Move the current piece
        
        Args:
            dx: Horizontal movement (-1: left, 1: right, 0: no movement)
            dy: Vertical movement (1: down, 0: no movement)
        
        Returns:
            True if movement was successful, False otherwise
        """
        new_x = self.current_x + dx
        new_y = self.current_y + dy
        
        if self._is_valid_position(new_x, new_y, self.current_rotation):
            self.current_x = new_x
            self.current_y = new_y
            return True
        return False
    
    def rotate_piece(self) -> bool:
        """
        Rotate the current piece clockwise
        
        Returns:
            True if rotation was successful, False otherwise
        """
        new_rotation = (self.current_rotation + 1) % 4
        
        if self._is_valid_position(self.current_x, self.current_y, new_rotation):
            self.current_rotation = new_rotation
            return True
        return False
    
    def drop_piece(self) -> int:
        """
        Drop the current piece to the bottom
        
        Returns:
            Number of lines cleared
        """
        # Move piece down until it can't move anymore
        while self.move_piece(0, 1):
            pass
        
        # Lock the piece in place
        self._lock_piece()
        
        # Clear completed lines
        lines_cleared = self._clear_lines()
        
        # Spawn new piece
        self._spawn_piece()
        
        return lines_cleared
    
    def _lock_piece(self) -> None:
        """Lock the current piece in place on the board"""
        shape = self._get_piece_shape(self.current_piece, self.current_rotation)
        
        for row in range(4):
            for col in range(4):
                if shape[row, col] == 1:
                    board_x = self.current_x + col
                    board_y = self.current_y + row
                    
                    if 0 <= board_y < self.board_height and 0 <= board_x < self.board_width:
                        self.board[board_y, board_x] = 1
    
    def _clear_lines(self) -> int:
        """Clear completed lines and return number of lines cleared"""
        lines_cleared = 0
        
        for row in range(self.board_height):
            if np.all(self.board[row] == 1):
                # Remove the line
                self.board = np.vstack([np.zeros((1, self.board_width)), 
                                       self.board[:row], 
                                       self.board[row+1:]])
                lines_cleared += 1
        
        # Update statistics
        self.lines_cleared += lines_cleared
        self.score += self._calculate_line_score(lines_cleared)
        self.level = self.lines_cleared // 10 + 1
        
        return lines_cleared
    
    def _calculate_line_score(self, lines_cleared: int) -> int:
        """Calculate score for clearing lines"""
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 100 * self.level
        elif lines_cleared == 2:
            return 300 * self.level
        elif lines_cleared == 3:
            return 500 * self.level
        else:  # Tetris (4 lines)
            return 800 * self.level
    
    def get_board_state(self) -> np.ndarray:
        """Get the current board state with the current piece"""
        board_with_piece = self.board.copy()
        
        if self.current_piece is not None and not self.game_over:
            shape = self._get_piece_shape(self.current_piece, self.current_rotation)
            
            for row in range(4):
                for col in range(4):
                    if shape[row, col] == 1:
                        board_x = self.current_x + col
                        board_y = self.current_y + row
                        
                        if (0 <= board_y < self.board_height and 
                            0 <= board_x < self.board_width):
                            board_with_piece[board_y, board_x] = 2  # Current piece
        
        return board_with_piece
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get complete game state"""
        return {
            'board': self.board.copy(),
            'board_with_piece': self.get_board_state(),
            'current_piece': self.current_piece,
            'current_x': self.current_x,
            'current_y': self.current_y,
            'current_rotation': self.current_rotation,
            'next_piece': self.next_piece,
            'lines_cleared': self.lines_cleared,
            'score': self.score,
            'level': self.level,
            'game_over': self.game_over
        }
    
    def reset(self) -> None:
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.lines_cleared = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.current_piece = None
        self.next_piece = None
        self._spawn_piece()
    
    def get_holes(self) -> int:
        """Count the number of holes in the board"""
        holes = 0
        for col in range(self.board_width):
            found_block = False
            for row in range(self.board_height):
                if self.board[row, col] == 1:
                    found_block = True
                elif found_block and self.board[row, col] == 0:
                    holes += 1
        return holes
    
    def get_height(self) -> int:
        """Get the height of the highest block"""
        for row in range(self.board_height):
            if np.any(self.board[row] == 1):
                return self.board_height - row
        return 0
    
    def get_roughness(self) -> int:
        """Calculate the roughness (height differences between adjacent columns)"""
        heights = []
        for col in range(self.board_width):
            for row in range(self.board_height):
                if self.board[row, col] == 1:
                    heights.append(self.board_height - row)
                    break
            else:
                heights.append(0)
        
        roughness = 0
        for i in range(len(heights) - 1):
            roughness += abs(heights[i] - heights[i + 1])
        
        return roughness
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the game state
        
        Args:
            mode: Rendering mode ('human' for ASCII, 'rgb_array' for RGB array)
        """
        if mode == 'human':
            # Print ASCII representation
            board_with_piece = self.get_board_state()
            print("=" * (self.board_width + 2))
            for row in board_with_piece:
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
            print(f"Score: {self.score}, Lines: {self.lines_cleared}")
        
        elif mode == 'rgb_array':
            # Return RGB array (same as environment)
            board_with_piece = self.get_board_state()
            rgb_board = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
            
            # Color mapping: 0=black, 1=gray, 2=blue
            rgb_board[board_with_piece == 0] = [0, 0, 0]      # Empty
            rgb_board[board_with_piece == 1] = [128, 128, 128] # Placed pieces
            rgb_board[board_with_piece == 2] = [0, 0, 255]     # Current piece
            
            return rgb_board 