"""
Improved Tetris Environment with better game mechanics
Optimized for longer games and better learning
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional

class ImprovedTetrisEnv:
    """
    Improved Tetris environment with better game mechanics
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
    
    def __init__(self, height: int = 20, width: int = 10, max_steps: int = 1000):
        self.height = height
        self.width = width
        self.max_steps = max_steps  # Add maximum steps to prevent infinite games
        self.grid = np.zeros((height, width), dtype=int)
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.current_rotation = 0
        self.lines_cleared = 0
        self.score = 0
        self.game_over = False
        self.pieces_placed = 0
        self.steps_taken = 0
        
        # Piece sequence (for reproducibility)
        self.piece_sequence = []
        self.sequence_index = 0
        
        # Game state tracking
        self.consecutive_no_actions = 0
        self.max_consecutive_no_actions = 5  # Allow some consecutive failures
        
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.lines_cleared = 0
        self.score = 0
        self.game_over = False
        self.pieces_placed = 0
        self.sequence_index = 0
        self.steps_taken = 0
        self.consecutive_no_actions = 0
        
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
        """Spawn a new piece at the top center with improved logic"""
        if self.sequence_index >= len(self.piece_sequence):
            self._generate_piece_sequence()
        
        piece_name = self.piece_sequence[self.sequence_index]
        self.current_piece = piece_name
        self.current_rotation = 0
        self.current_x = self.width // 2 - 1
        self.current_y = 0
        
        # Improved game over check: only end if no valid actions exist
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            self.consecutive_no_actions += 1
            if self.consecutive_no_actions >= self.max_consecutive_no_actions:
                self.game_over = True
        else:
            self.consecutive_no_actions = 0  # Reset counter if we have valid actions
    
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
    
    def get_final_landing_positions(self) -> List[Tuple[int, int]]:
        """
        Get all possible final landing positions (x_position, rotation)
        Only returns positions where the piece can land at the bottom
        """
        landing_positions = []
        
        for rotation in range(4):  # 4 possible rotations
            shape = self._get_piece_shape(self.current_piece, rotation)
            piece_width = shape.shape[1]
            
            for x in range(self.width - piece_width + 1):
                # Check if this position is valid for landing
                if self._can_land_at_position(x, rotation):
                    landing_positions.append((x, rotation))
        
        return landing_positions
    
    def _can_land_at_position(self, x: int, rotation: int) -> bool:
        """
        Check if piece can land at the given position
        Returns True if the piece can be placed at the bottom at this position
        """
        shape = self._get_piece_shape(self.current_piece, rotation)
        piece_height, piece_width = shape.shape
        
        # Check if position is within bounds
        if x < 0 or x + piece_width > self.width:
            return False
        
        # Find the lowest possible landing position
        landing_y = self.height - piece_height  # Start from bottom
        
        # Move up until we find a valid landing position
        while landing_y >= 0:
            # Check if this position is valid (no collision)
            valid = True
            for y in range(piece_height):
                for x_offset in range(piece_width):
                    if shape[y, x_offset] == 1:
                        grid_y = landing_y + y
                        grid_x = x + x_offset
                        
                        # Check bounds
                        if grid_y >= self.height or grid_x >= self.width:
                            valid = False
                            break
                        
                        # Check collision with existing pieces
                        if grid_y >= 0 and self.grid[grid_y, grid_x] == 1:
                            valid = False
                            break
                
                if not valid:
                    break
            
            if valid:
                # Check if there's a piece directly below (i.e., this is a landing position)
                has_support = False
                for y in range(piece_height):
                    for x_offset in range(piece_width):
                        if shape[y, x_offset] == 1:
                            grid_y = landing_y + y
                            grid_x = x + x_offset
                            
                            # Check if there's a piece directly below
                            if grid_y + 1 < self.height and self.grid[grid_y + 1, grid_x] == 1:
                                has_support = True
                                break
                    
                    if has_support:
                        break
                
                # If we're at the bottom or have support, this is a valid landing position
                if landing_y + piece_height >= self.height or has_support:
                    return True
                else:
                    # Move up one position and try again
                    landing_y -= 1
            else:
                # Move up one position and try again
                landing_y -= 1
        
        return False
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        Take an action (x_position, rotation)
        Returns: (state, reward, done)
        """
        if self.game_over:
            return self._get_state(), 0, True
        
        self.steps_taken += 1
        
        # Check if we've exceeded max steps
        if self.steps_taken >= self.max_steps:
            self.game_over = True
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
        
        # Calculate reward (improved: consider both lines and game length)
        lines_cleared_this_step = self.lines_cleared - (self.pieces_placed - 1)
        reward = lines_cleared_this_step + 0.1  # Small reward for surviving
        
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
                        if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                            display_grid[grid_y, grid_x] = 2  # Different value for current piece
        
        # Print the grid
        for row in display_grid:
            print(''.join(['█' if cell == 1 else '░' if cell == 2 else ' ' for cell in row]))
        
        print(f"Lines: {self.lines_cleared}, Score: {self.score}, Steps: {self.steps_taken}")
        print(f"Pieces: {self.pieces_placed}, Game Over: {self.game_over}") 