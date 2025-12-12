#!/usr/bin/env python3
"""
Simplified DQN training - no complex callbacks, just basic training.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


class GymnasiumWrapper(gym.Env):
    """Wrapper to convert old Gym environment to Gymnasium."""
    
    def __init__(self, load_pattern='SINE_CURVE'):
        super().__init__()
        self.env = ScalingEnv()
        self.env.scaling_env_options['input'] = INPUTS[load_pattern]
        self.env.change_rate = 1
        
        self.action_space = spaces.Discrete(self.env.num_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs.astype(np.float32), {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.astype(np.float32), reward, done, False, info
    
    def close(self):
        return self.env.close()
    
    @property
    def total_cost(self):
        return self.env.total_cost
    
    @property
    def instances(self):
        return self.env.instances
    
    @property
    def load(self):
        return self.env.load
    
    @property
    def queue_size(self):
        return self.env.queue_size
    
    @property
    def influx(self):
        return self.env.influx
    
    @property
    def actions(self):
        return self.env.actions


def train_dqn_simple(timesteps=100000, load_pattern='SINE_CURVE'):
    """Simple DQN training without complex callbacks."""
    
    # Check GPU
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dqn_simple_{load_pattern.lower()}_{timestamp}"
    
    print("=" * 70)
    print("DQN Training - Simplified Version")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Load pattern: {load_pattern}")
    print(f"Total timesteps: {timesteps:,}")
    print(f"Model name: {model_name}")
    print("=" * 70)
    
    # Create environment
    env = GymnasiumWrapper(load_pattern)
    env = Monitor(env)
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=42,
        device=device
    )
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Train
    print("\nStarting training...")
    print("(Progress will be shown every 100 steps)\n")
    
    start_time = datetime.now()
    
    # Simple training with built-in progress bar
    model.learn(
        total_timesteps=timesteps,
        log_interval=100,
        progress_bar=True
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    # Save model
    model_path = f"models/{model_name}.zip"
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Training duration: {training_duration/60:.2f} minutes")
    print(f"Timesteps per second: {timesteps/training_duration:.2f}")
    print("=" * 70)
    
    env.close()
    
    return model, model_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN (simplified version)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'RANDOM', 'STEADY'],
                       help='Load pattern for training')
    
    args = parser.parse_args()
    
    # Train
    model, model_path = train_dqn_simple(
        timesteps=args.timesteps,
        load_pattern=args.pattern
    )
    
    print("\n✓ Training completed successfully!")
