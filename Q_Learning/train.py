"""
Training script for Tetris Q-Learning agent
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from src.environment import TetrisEnv
from src.agents import QLearningAgent


def train_episode(env, agent):
    """
    Train for one episode
    
    Args:
        env: Tetris environment
        agent: Q-Learning agent
        
    Returns:
        episode_reward, episode_steps, episode_lines
    """
    observation, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_lines = 0
    
    while True:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Choose action
        action = agent.get_action(observation, valid_actions)
        
        # Take step
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Get next valid actions
        next_valid_actions = env.get_valid_actions()
        
        # Update agent
        agent.update(observation, action, reward, next_observation, 
                    next_valid_actions, terminated or truncated)
        
        # Update episode statistics
        episode_reward += reward
        episode_steps += 1
        episode_lines = info['lines_cleared']
        
        # Move to next state
        observation = next_observation
        
        # Check if episode is done
        if terminated or truncated:
            break
    
    return episode_reward, episode_steps, episode_lines


def train(episodes=10000, save_interval=1000, model_dir='data/models'):
    """
    Train the Q-Learning agent
    
    Args:
        episodes: Number of episodes to train
        save_interval: Save model every N episodes
        model_dir: Directory to save models
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = TetrisEnv()
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )
    
    # Training statistics
    training_history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_lines': [],
        'epsilon_values': [],
        'q_table_sizes': []
    }
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon}")
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        # Train one episode
        reward, steps, lines = train_episode(env, agent)
        
        # Update agent statistics
        agent.update_training_stats(reward, steps, lines)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record training history
        training_history['episode_rewards'].append(reward)
        training_history['episode_lengths'].append(steps)
        training_history['episode_lines'].append(lines)
        training_history['epsilon_values'].append(agent.epsilon)
        training_history['q_table_sizes'].append(agent.get_q_table_size())
        
        # Print progress
        if (episode + 1) % 100 == 0:
            stats = agent.get_training_stats()
            recent_rewards = training_history['episode_rewards'][-100:]
            recent_lines = training_history['episode_lines'][-100:]
            
            print(f"Episode {episode + 1}: "
                  f"Avg Reward = {np.mean(recent_rewards):.2f}, "
                  f"Avg Lines = {np.mean(recent_lines):.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}, "
                  f"Q-table size = {agent.get_q_table_size()}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f'q_learning_episode_{episode + 1}.pkl')
            agent.save_q_table(model_path)
            
            # Save training history
            history_path = os.path.join(model_dir, f'training_history_episode_{episode + 1}.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f)
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'q_learning_final.pkl')
    agent.save_q_table(final_model_path)
    
    # Save final training history
    final_history_path = os.path.join(model_dir, 'training_history_final.json')
    with open(final_history_path, 'w') as f:
        json.dump(training_history, f)
    
    print(f"Training completed!")
    print(f"Final Q-table size: {agent.get_q_table_size()}")
    print(f"Final epsilon: {agent.epsilon}")
    
    return agent, training_history


def plot_training_results(training_history, save_dir='data/plots'):
    """
    Plot training results
    
    Args:
        training_history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tetris Q-Learning Training Results', fontsize=16)
    
    # Plot episode rewards
    axes[0, 0].plot(training_history['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot episode lengths
    axes[0, 1].plot(training_history['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Plot lines cleared
    axes[1, 0].plot(training_history['episode_lines'])
    axes[1, 0].set_title('Lines Cleared per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Lines')
    axes[1, 0].grid(True)
    
    # Plot epsilon decay
    axes[1, 1].plot(training_history['epsilon_values'])
    axes[1, 1].set_title('Epsilon Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'training_results_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved to: {plot_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Tetris Q-Learning agent')
    parser.add_argument('--episodes', type=int, default=10000, 
                       help='Number of episodes to train')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save model every N episodes')
    parser.add_argument('--model_dir', type=str, default='data/models',
                       help='Directory to save models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Train agent
    agent, training_history = train(
        episodes=args.episodes,
        save_interval=args.save_interval,
        model_dir=args.model_dir
    )
    
    # Plot results if requested
    if args.plot:
        plot_training_results(training_history)


if __name__ == "__main__":
    main() 