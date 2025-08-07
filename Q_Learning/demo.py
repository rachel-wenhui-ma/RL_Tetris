"""
Demo script to showcase the Tetris RL project
"""

import numpy as np
import time
from src.environment import TetrisEnv, TetrisGame
from src.agents import QLearningAgent


def demo_tetris_game():
    """Demonstrate basic Tetris game functionality"""
    print("🎮 Tetris Game Demo")
    print("=" * 50)
    
    game = TetrisGame()
    
    print(f"Board size: {game.board_width}x{game.board_height}")
    print(f"Current piece: {game.current_piece}")
    print(f"Next piece: {game.next_piece}")
    print(f"Score: {game.score}")
    print(f"Lines cleared: {game.lines_cleared}")
    print(f"Level: {game.level}")
    
    # Show initial board
    print("\nInitial board:")
    game.render(mode='human')
    
    # Play a few moves
    print("\nPlaying a few moves...")
    for i in range(5):
        print(f"\nMove {i+1}:")
        
        # Rotate piece
        if game.rotate_piece():
            print("✓ Rotated piece")
        
        # Move piece
        if game.move_piece(1, 0):
            print("✓ Moved piece right")
        
        # Drop piece
        lines_cleared = game.drop_piece()
        if lines_cleared > 0:
            print(f"✓ Cleared {lines_cleared} lines!")
        
        print(f"Score: {game.score}, Lines: {game.lines_cleared}")
        
        if game.game_over:
            print("Game over!")
            break
    
    print("\n" + "=" * 50)


def demo_tetris_env():
    """Demonstrate Tetris environment"""
    print("🤖 Tetris Environment Demo")
    print("=" * 50)
    
    env = TetrisEnv()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    observation, info = env.reset()
    print(f"\nInitial observation keys: {list(observation.keys())}")
    print(f"Board shape: {observation['board'].shape}")
    print(f"Current piece: {observation['current_piece']}")
    print(f"Next piece: {observation['next_piece']}")
    
    # Take a few steps
    print("\nTaking a few steps...")
    for i in range(3):
        print(f"\nStep {i+1}:")
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        print(f"Valid actions: {len(valid_actions)} available")
        
        # Choose random action
        action = valid_actions[0] if valid_actions else (0, 5)
        print(f"Chosen action: {action}")
        
        # Take step
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Info: {info}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print("\n" + "=" * 50)


def demo_q_learning_agent():
    """Demonstrate Q-Learning agent"""
    print("🧠 Q-Learning Agent Demo")
    print("=" * 50)
    
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    print(f"Learning rate: {agent.lr}")
    print(f"Discount factor: {agent.gamma}")
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Initial Q-table size: {agent.get_q_table_size()}")
    
    # Create a simple state
    state = {
        'board': np.zeros((20, 10), dtype=np.int8),
        'current_piece': 0,
        'next_piece': 1
    }
    valid_actions = [(0, 5), (1, 5), (2, 5)]
    
    # Test action selection
    action = agent.get_action(state, valid_actions)
    print(f"\nSelected action: {action}")
    print(f"Action is valid: {action in valid_actions}")
    
    # Test Q-value update
    next_state = {
        'board': np.zeros((20, 10), dtype=np.int8),
        'current_piece': 1,
        'next_piece': 2
    }
    next_valid_actions = [(0, 5), (1, 5)]
    
    print(f"\nUpdating Q-values...")
    agent.update(state, action, 10.0, next_state, next_valid_actions, False)
    
    print(f"Q-table size after update: {agent.get_q_table_size()}")
    print(f"Total Q-values: {agent.get_total_q_values()}")
    
    # Test epsilon decay
    print(f"\nEpsilon before decay: {agent.epsilon}")
    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon}")
    
    print("\n" + "=" * 50)


def demo_training_episode():
    """Demonstrate a single training episode"""
    print("🎯 Training Episode Demo")
    print("=" * 50)
    
    env = TetrisEnv()
    agent = QLearningAgent(epsilon=0.1)
    
    print("Starting training episode...")
    
    observation, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_lines = 0
    
    while episode_steps < 50:  # Limit for demo
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Choose action
        action = agent.get_action(observation, valid_actions)
        
        # Take step
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # Update agent
        agent.update(observation, action, reward, next_observation, 
                    next_valid_actions, terminated or truncated)
        
        # Update statistics
        episode_reward += reward
        episode_steps += 1
        episode_lines = info['lines_cleared']
        
        # Move to next state
        observation = next_observation
        
        # Print progress every 10 steps
        if episode_steps % 10 == 0:
            print(f"Step {episode_steps}: Reward={episode_reward:.1f}, Lines={episode_lines}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed!")
    print(f"Total steps: {episode_steps}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Lines cleared: {episode_lines}")
    print(f"Q-table size: {agent.get_q_table_size()}")
    
    # Update agent statistics
    agent.update_training_stats(episode_reward, episode_steps, episode_lines)
    stats = agent.get_training_stats()
    print(f"Training stats: {stats}")
    
    print("\n" + "=" * 50)


def demo_visualization():
    """Demonstrate visualization features"""
    print("🎨 Visualization Demo")
    print("=" * 50)
    
    env = TetrisEnv()
    observation, info = env.reset()
    
    print("ASCII rendering:")
    env.render(mode='human')
    
    print("\nRGB array rendering:")
    rgb_array = env.render(mode='rgb_array')
    print(f"RGB array shape: {rgb_array.shape}")
    print(f"RGB array dtype: {rgb_array.dtype}")
    print(f"Unique colors: {np.unique(rgb_array)}")
    
    # Take a few steps and show progression
    print("\nProgression demo:")
    for i in range(3):
        print(f"\nStep {i+1}:")
        valid_actions = env.get_valid_actions()
        action = valid_actions[0] if valid_actions else (0, 5)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render(mode='human')
        time.sleep(1)  # Pause to show progression
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 50)


def main():
    """Run all demos"""
    print("🚀 Tetris Reinforcement Learning Project Demo")
    print("=" * 60)
    
    try:
        demo_tetris_game()
        demo_tetris_env()
        demo_q_learning_agent()
        demo_training_episode()
        demo_visualization()
        
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python test_basic.py' to verify everything works")
        print("2. Run 'python train.py --episodes 1000' to start training")
        print("3. Check the README.md for more information")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 