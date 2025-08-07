"""
Basic test script to verify the Tetris environment and agent work correctly
"""

import numpy as np
from src.environment import TetrisEnv, TetrisGame
from src.agents import QLearningAgent


def test_tetris_game():
    """Test the basic Tetris game functionality"""
    print("Testing Tetris game...")
    
    game = TetrisGame()
    
    # Test initial state
    assert game.board.shape == (20, 10), f"Board shape should be (20, 10), got {game.board.shape}"
    assert game.current_piece is not None, "Current piece should not be None"
    assert not game.game_over, "Game should not be over initially"
    
    # Test piece movement
    initial_x = game.current_x
    success = game.move_piece(1, 0)  # Move right
    assert success, "Moving right should succeed"
    assert game.current_x == initial_x + 1, "X position should increase by 1"
    
    # Test piece rotation
    initial_rotation = game.current_rotation
    success = game.rotate_piece()
    assert success, "Rotation should succeed"
    assert game.current_rotation == (initial_rotation + 1) % 4, "Rotation should increase by 1"
    
    # Test piece dropping
    lines_cleared = game.drop_piece()
    assert isinstance(lines_cleared, int), "Lines cleared should be an integer"
    assert lines_cleared >= 0, "Lines cleared should be non-negative"
    
    print("✓ Tetris game tests passed!")


def test_tetris_env():
    """Test the Tetris environment"""
    print("Testing Tetris environment...")
    
    env = TetrisEnv()
    
    # Test reset
    observation, info = env.reset()
    assert isinstance(observation, dict), "Observation should be a dictionary"
    assert 'board' in observation, "Observation should contain 'board'"
    assert observation['board'].shape == (20, 10), "Board should be 20x10"
    
    # Test action space
    assert env.action_space.shape == (2,), "Action space should have shape (2,)"
    assert env.action_space.nvec[0] == 4, "Rotation should have 4 possible values"
    assert env.action_space.nvec[1] == 11, "Move should have 11 possible values"
    
    # Test step
    action = (0, 5)  # No rotation, center position
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_observation, dict), "Next observation should be a dictionary"
    assert isinstance(reward, (int, float)), "Reward should be a number"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    
    # Test valid actions
    valid_actions = env.get_valid_actions()
    assert isinstance(valid_actions, list), "Valid actions should be a list"
    assert len(valid_actions) > 0, "Should have at least one valid action"
    
    print("✓ Tetris environment tests passed!")


def test_q_learning_agent():
    """Test the Q-Learning agent"""
    print("Testing Q-Learning agent...")
    
    agent = QLearningAgent()
    
    # Test action selection
    state = {
        'board': np.zeros((20, 10), dtype=np.int8),
        'current_piece': 0,
        'next_piece': 1
    }
    valid_actions = [(0, 5), (1, 5), (2, 5)]
    
    action = agent.get_action(state, valid_actions)
    assert action in valid_actions, "Selected action should be valid"
    
    # Test Q-value update
    next_state = {
        'board': np.zeros((20, 10), dtype=np.int8),
        'current_piece': 1,
        'next_piece': 2
    }
    next_valid_actions = [(0, 5), (1, 5)]
    
    agent.update(state, action, 10.0, next_state, next_valid_actions, False)
    
    # Check that Q-table was updated
    state_key = agent._state_to_key(state)
    assert state_key in agent.q_table, "State should be in Q-table after update"
    assert action in agent.q_table[state_key], "Action should be in Q-table after update"
    
    print("✓ Q-Learning agent tests passed!")


def test_training_episode():
    """Test a single training episode"""
    print("Testing training episode...")
    
    env = TetrisEnv()
    agent = QLearningAgent(epsilon=0.0)  # No exploration for testing
    
    observation, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    while episode_steps < 100:  # Limit steps for testing
        valid_actions = env.get_valid_actions()
        action = agent.get_action(observation, valid_actions)
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        agent.update(observation, action, reward, next_observation, 
                    next_valid_actions, terminated or truncated)
        
        episode_reward += reward
        episode_steps += 1
        observation = next_observation
        
        if terminated or truncated:
            break
    
    assert episode_steps > 0, "Episode should have at least one step"
    assert isinstance(episode_reward, (int, float)), "Episode reward should be a number"
    
    print(f"✓ Training episode completed: {episode_steps} steps, {episode_reward:.2f} reward")


def test_visualization():
    """Test environment visualization"""
    print("Testing visualization...")
    
    env = TetrisEnv()
    observation, info = env.reset()
    
    # Test ASCII rendering
    env.render(mode='human')
    
    # Test RGB array rendering
    rgb_array = env.render(mode='rgb_array')
    assert rgb_array is not None, "RGB array should not be None"
    assert rgb_array.shape == (20, 10, 3), f"RGB array shape should be (20, 10, 3), got {rgb_array.shape}"
    assert rgb_array.dtype == np.uint8, "RGB array should be uint8"
    
    print("✓ Visualization tests passed!")


def main():
    """Run all tests"""
    print("Running basic tests...\n")
    
    try:
        test_tetris_game()
        test_tetris_env()
        test_q_learning_agent()
        test_training_episode()
        test_visualization()
        
        print("\n🎉 All tests passed! The code is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 