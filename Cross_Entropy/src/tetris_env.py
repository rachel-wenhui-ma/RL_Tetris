"""
Tetris Environment for Paper Reproduction
Based on standard Tetris rules used in research papers
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional

class TetrisEnv:
    """
    Standard Tetris environment as used in research papers
    Grid size: 20x10 (height x width)
    """
    
    # Tetromino shapes (I, O, T, S, Z, J, L)
    TETROMINOS = {
        'I': [[1, 1, 1, 1]],
        'O': [[1, 1], [1, 1]],
        'T': [[0, 1, 0], [1, 1, 1]],
        'S': [[0, 1, 1], [1, 1, 0]],
        'Z': [[1, 1, 0], [0, 1, 1]],
        'J': [[1, 0, 0], [1, 1, 1]],
        'L': [[0, 0, 1], [1, 1, 1]]
    }
    
    def __init__(self, height: int = 20, width: int = 10):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.current_rotation = 0
        self.lines_cleared = 0
        self.score = 0
        self.game_over = False
        self.pieces_placed = 0
        
        # Piece sequence (for reproducibility)
        self.piece_sequence = []
        self.sequence_index = 0
        
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.lines_cleared = 0
        self.score = 0
        self.game_over = False
        self.pieces_placed = 0
        self.sequence_index = 0
        
        # Generate new piece sequence
        self._generate_piece_sequence()
        
        # Spawn first piece
        self._spawn_piece()
        
        return self._get_state()
    
    def _generate_piece_sequence(self, length: int = 1000):
        """Generate a sequence of pieces for reproducibility"""
        pieces = list(self.TETROMINOS.keys())
        self.piece_sequence = []
        for _ in range(length):
            # Generate 7 pieces in random order (like real Tetris)
            bag = pieces.copy()
            random.shuffle(bag)
            self.piece_sequence.extend(bag)
    
    def _spawn_piece(self):
        """Spawn a new piece at the top center"""
        if self.sequence_index >= len(self.piece_sequence):
            self._generate_piece_sequence()
        
        piece_name = self.piece_sequence[self.sequence_index]
        self.current_piece = piece_name
        self.current_rotation = 0
        self.current_x = self.width // 2 - 1
        self.current_y = 0
        
        # Check if game is over
        if self._check_collision():
            self.game_over = True
    
    def _get_piece_shape(self, piece: str, rotation: int = 0) -> np.ndarray:
        """Get the shape of a piece with given rotation"""
        shape = np.array(self.TETROMINOS[piece])
        
        # Apply rotation
        for _ in range(rotation):
            shape = np.rot90(shape)
        
        return shape
    
    def _check_collision(self, dx: int = 0, dy: int = 0, rotation: int = None) -> bool:
        """Check if current piece collides with walls or other pieces"""
        if rotation is None:
            rotation = self.current_rotation
            
        shape = self._get_piece_shape(self.current_piece, rotation)
        piece_height, piece_width = shape.shape
        
        new_x = self.current_x + dx
        new_y = self.current_y + dy
        
        # Check boundaries
        if (new_x < 0 or new_x + piece_width > self.width or 
            new_y + piece_height > self.height):
            return True
        
        # Check collision with placed pieces
        for y in range(piece_height):
            for x in range(piece_width):
                if shape[y, x] == 1:
                    grid_y = new_y + y
                    grid_x = new_x + x
                    if grid_y >= 0 and self.grid[grid_y, grid_x] == 1:
                        return True
        
        return False
    
    def _place_piece(self):
        """Place the current piece on the grid"""
        shape = self._get_piece_shape(self.current_piece, self.current_rotation)
        piece_height, piece_width = shape.shape
        
        for y in range(piece_height):
            for x in range(piece_width):
                if shape[y, x] == 1:
                    grid_y = self.current_y + y
                    grid_x = self.current_x + x
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        self.grid[grid_y, grid_x] = 1
        
        self.pieces_placed += 1
        self._clear_lines()
        self.sequence_index += 1
        self._spawn_piece()
    
    def _clear_lines(self):
        """Clear completed lines and update score"""
        lines_to_clear = []
        
        for y in range(self.height):
            if np.all(self.grid[y, :] == 1):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # Remove completed lines
            for line in reversed(lines_to_clear):
                self.grid = np.delete(self.grid, line, axis=0)
                self.grid = np.vstack([np.zeros((1, self.width), dtype=int), self.grid])
            
            # Update score and lines cleared
            lines_cleared = len(lines_to_clear)
            self.lines_cleared += lines_cleared
            self.score += lines_cleared  # Simple scoring: 1 point per line
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get all valid actions (x_position, rotation) for current piece"""
        valid_actions = []
        
        for rotation in range(4):  # 4 possible rotations
            shape = self._get_piece_shape(self.current_piece, rotation)
            piece_width = shape.shape[1]
            
            for x in range(self.width - piece_width + 1):
                # Check if this position is valid
                if not self._check_collision(dx=x-self.current_x, dy=0, rotation=rotation):
                    valid_actions.append((x, rotation))
        
        return valid_actions
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        Take an action (x_position, rotation)
        Returns: (state, reward, done)
        """
        if self.game_over:
            return self._get_state(), 0, True
        
        x_pos, rotation = action
        
        # Move piece to target position
        self.current_x = x_pos
        self.current_rotation = rotation
        
        # Drop piece to bottom
        while not self._check_collision(dy=1):
            self.current_y += 1
        
        # Place piece
        self._place_piece()
        
        # Calculate reward (simple: lines cleared)
        reward = self.score - (self.pieces_placed - 1)  # Reward for lines cleared
        
        return self._get_state(), reward, self.game_over
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        return self.grid.copy()
    
    def get_height_profile(self) -> np.ndarray:
        """Get height of each column"""
        heights = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[y, x] == 1:
                    heights.append(self.height - y)
                    break
            else:
                heights.append(0)
        return np.array(heights)
    
    def get_holes(self) -> int:
        """Count number of holes (empty cells with filled cells above)"""
        holes = 0
        for x in range(self.width):
            found_filled = False
            for y in range(self.height):
                if self.grid[y, x] == 1:
                    found_filled = True
                elif found_filled and self.grid[y, x] == 0:
                    holes += 1
        return holes
    
    def get_wells(self) -> List[int]:
        """Get well depths (empty columns with filled cells on both sides)"""
        wells = []
        for x in range(self.width):
            well_depth = 0
            for y in range(self.height):
                if self.grid[y, x] == 0:
                    # Check if this is a well (empty with filled on sides)
                    left_filled = (x > 0 and self.grid[y, x-1] == 1)
                    right_filled = (x < self.width-1 and self.grid[y, x+1] == 1)
                    if left_filled and right_filled:
                        well_depth += 1
                    else:
                        break
                else:
                    break
            wells.append(well_depth)
        return wells
    
    def render(self):
        """Render the current state"""
        display_grid = self.grid.copy()
        
        # Add current piece to display
        if not self.game_over and self.current_piece:
            shape = self._get_piece_shape(self.current_piece, self.current_rotation)
            piece_height, piece_width = shape.shape
            
            for y in range(piece_height):
                for x in range(piece_width):
                    if shape[y, x] == 1:
                        grid_y = self.current_y + y
                        grid_x = self.current_x + x
                        if (0 <= grid_y < self.height and 
                            0 <= grid_x < self.width):
                            display_grid[grid_y, grid_x] = 2
        
        # Print grid
        print("=" * (self.width * 2 + 1))
        for row in display_grid:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("  ", end="")
                elif cell == 1:
                    print("██", end="")
                else:
                    print("[]", end="")
            print("|")
        print("=" * (self.width * 2 + 1))
        print(f"Score: {self.score}, Lines: {self.lines_cleared}, Pieces: {self.pieces_placed}") 