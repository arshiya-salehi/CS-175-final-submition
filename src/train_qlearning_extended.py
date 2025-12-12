#!/usr/bin/env python3
"""
Extended Q-Learning implementation for cloud autoscaling.
Optimized for longer training runs (10,000+ episodes).
"""

import numpy as np
import random
import pickle
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


class QLearningAgent:
    """Tabular Q-Learning agent with discretized state space."""
    
    def __init__(self, n_actions=3, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: defaultdict returns 0 for unseen state-action pairs
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def discretize_state(self, obs, bins=10):
        """
        Discretize continuous observation into discrete state.
        
        Args:
            obs: [normalized_instances, load, capacity, influx, queue_size]
            bins: Number of bins per dimension
        
        Returns:
            Tuple representing discrete state
        """
        # Focus on most important features
        instances_bin = min(int(obs[0] * bins), bins - 1)
        load_bin = min(int(obs[1] * bins), bins - 1)
        
        # Discretize queue size (0, small, medium, large)
        if obs[4] == 0:
            queue_bin = 0
        elif obs[4] < 100:
            queue_bin = 1
        elif obs[4] < 500:
            queue_bin = 2
        else:
            queue_bin = 3
        
        return (instances_bin, load_bin, queue_bin)
    
    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Discrete state tuple
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), pickle.load(f))
        print(f"✓ Model loaded from {filepath}")


def train_qlearning_extended(n_episodes=10000, max_steps=200, verbose=True, model_name=None):
    """
    Train Q-learning agent on autoscaling environment with extended episodes.
    
    Args:
        n_episodes: Number of training episodes (default 10,000)
        max_steps: Maximum steps per episode
        verbose: Print progress
        model_name: Custom model name
    
    Returns:
        Trained agent and training metrics
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"qlearning_extended_{timestamp}"
    
    # Create environment
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    
    # Create agent with slower epsilon decay for longer training
    agent = QLearningAgent(
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995  # Slower decay for 10k episodes
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    episode_queues = []
    episode_loads = []
    
    print("=" * 70)
    print("Training Q-Learning Agent (Extended)")
    print("=" * 70)
    print(f"Episodes: {n_episodes:,}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Learning rate: {agent.lr}")
    print(f"Discount factor: {agent.gamma}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"Model name: {model_name}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Create progress bar
    pbar = tqdm(range(n_episodes), desc="Training Q-Learning", unit="episode", dynamic_ncols=True)
    
    for episode in pbar:
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        episode_cost = 0
        queues = []
        loads = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.get_action(state, training=True)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            # Track metrics
            episode_reward += reward
            episode_cost = env.total_cost
            queues.append(env.queue_size)
            loads.append(env.load)
            
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        episode_costs.append(episode_cost)
        episode_queues.append(np.mean(queues))
        episode_loads.append(np.mean(loads))
        
        # Update progress bar
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_cost = np.mean(episode_costs[-100:])
            pbar.set_postfix({
                'reward': f'{avg_reward:.2f}',
                'cost': f'${avg_cost:.2f}',
                'ε': f'{agent.epsilon:.4f}'
            })
    
    pbar.close()
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print("=" * 70)
    print("Training completed!")
    print(f"✓ Total training time: {training_duration/60:.2f} minutes")
    print(f"✓ Episodes per second: {n_episodes/training_duration:.2f}")
    print(f"✓ Final epsilon: {agent.epsilon:.4f}")
    print(f"✓ Q-table size: {len(agent.q_table):,} states")
    print("=" * 70)
    
    # Save model
    model_path = f"models/{model_name}.pkl"
    agent.save(model_path)
    
    return agent, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'costs': episode_costs,
        'queues': episode_queues,
        'loads': episode_loads
    }


def evaluate_agent(agent, n_episodes=10, max_steps=200, load_pattern='SINE_CURVE'):
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained Q-learning agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        load_pattern: Load pattern to evaluate on
    
    Returns:
        Evaluation metrics
    """
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS[load_pattern]
    env.change_rate = 1
    
    episode_rewards = []
    episode_costs = []
    episode_queues = []
    episode_loads = []
    
    print("\n" + "=" * 70)
    print(f"Evaluating Q-Learning Agent on {load_pattern}")
    print("=" * 70)
    
    for episode in range(n_episodes):
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        queues = []
        loads = []
        
        for step in range(max_steps):
            # Greedy action selection (no exploration)
            action = agent.get_action(state, training=False)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            episode_reward += reward
            queues.append(env.queue_size)
            loads.append(env.load)
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(env.total_cost)
        episode_queues.append(np.mean(queues))
        episode_loads.append(np.mean(loads))
        
        print(f"  Episode {episode + 1:2d}: "
              f"Reward={episode_reward:7.2f}, "
              f"Cost=${env.total_cost:8.2f}, "
              f"Avg Queue={np.mean(queues):6.2f}, "
              f"Avg Load={np.mean(loads):5.1f}%")
    
    print("=" * 70)
    print("Evaluation Summary:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Cost: ${np.mean(episode_costs):.2f} ± ${np.std(episode_costs):.2f}")
    print(f"  Average Queue: {np.mean(episode_queues):.2f} ± {np.std(episode_queues):.2f}")
    print(f"  Average Load: {np.mean(episode_loads):.1f}% ± {np.std(episode_loads):.1f}%")
    print("=" * 70)
    
    return {
        'rewards': episode_rewards,
        'costs': episode_costs,
        'queues': episode_queues,
        'loads': episode_loads
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Q-Learning agent with extended episodes')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes (default: 10,000)')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for saving')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Train agent
    agent, train_metrics = train_qlearning_extended(
        n_episodes=args.episodes,
        max_steps=200,
        verbose=True,
        model_name=args.name
    )
    
    # Evaluate if requested
    if args.eval:
        eval_metrics = evaluate_agent(agent, n_episodes=10, max_steps=200)
