#!/usr/bin/env python3
"""
Train PPO agent using M4 GPU acceleration (Metal Performance Shaders).
Optimized for MacBook Air M4 chip.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
from gym_scaling.load_generators import LOAD_PATTERNS


class GymnasiumWrapper(gym.Env):
    """Wrapper to convert old Gym environment to Gymnasium."""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(env.num_actions)
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
    
    def render(self):
        return self.env.render()
    
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


class TqdmCallback(BaseCallback):
    """Custom callback for progress bar with tqdm."""
    
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO", 
                        unit="steps", dynamic_ncols=True)
        
    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()


def check_gpu_availability():
    """Check if GPU is available (MPS for Mac, CUDA for others)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("✓ MPS (Metal Performance Shaders) is available!")
        print(f"✓ Using M4 GPU acceleration")
        return "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA GPU is available!")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⚠️  No GPU found. Training will use CPU.")
        return "cpu"


def create_env(load_pattern='SINE_CURVE'):
    """Create environment with specified load pattern."""
    env = ScalingEnv()
    
    # Set load pattern
    if load_pattern in LOAD_PATTERNS:
        pattern = LOAD_PATTERNS[load_pattern]
        def load_func(step, max_influx, offset):
            return pattern['function'](step, max_influx, offset, **pattern['options'])
        env.scaling_env_options['input'] = {
            'function': load_func,
            'options': {}
        }
    elif load_pattern == 'SINE_CURVE':
        env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    else:
        env.scaling_env_options['input'] = INPUTS['RANDOM']
    
    env.change_rate = 1  # Change influx every step
    
    # Wrap in Gymnasium wrapper
    return GymnasiumWrapper(env)


def train_ppo_gpu(load_pattern='SINE_CURVE', total_timesteps=100000, model_name=None):
    """
    Train PPO agent with GPU acceleration.
    
    Args:
        load_pattern: Load pattern to use ('SINE_CURVE', 'SINUSOIDAL', 'SPIKE', etc.)
        total_timesteps: Total training timesteps (default 100k for extended training)
        model_name: Name for saved model (auto-generated if None)
    
    Returns:
        Trained model and save path
    """
    # Check GPU availability
    device = check_gpu_availability()
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_gpu_{load_pattern.lower()}_{timestamp}"
    
    print("=" * 70)
    print("Training PPO Agent with GPU Acceleration")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Load pattern: {load_pattern}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model name: {model_name}")
    print("=" * 70)
    
    # Create environment
    env = create_env(load_pattern)
    env = Monitor(env)  # Wrap for logging
    
    # Create PPO model with GPU support
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network for GPU
        verbose=1,
        seed=42,
        device=device  # Use GPU
    )
    
    # Set up callbacks
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('models/best', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints',
        name_prefix=model_name
    )
    
    # Create eval environment
    eval_env = create_env(load_pattern)
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/best',
        log_path='logs/ppo_eval',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create progress bar callback
    progress_callback = TqdmCallback(total_timesteps)
    
    # Train
    print("\n🚀 Starting training with GPU acceleration...")
    print(f"Progress bar will show training status\n")
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, progress_callback],
        log_interval=10
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    # Save final model
    model_path = f"models/{model_name}.zip"
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Training duration: {training_duration/60:.2f} minutes")
    print(f"✓ Timesteps per second: {total_timesteps/training_duration:.2f}")
    print("=" * 70)
    
    env.close()
    eval_env.close()
    
    return model, model_path


def evaluate_ppo(model_path, load_pattern='SINE_CURVE', num_episodes=10, max_steps=200):
    """
    Evaluate trained PPO model.
    
    Args:
        model_path: Path to saved model
        load_pattern: Load pattern to test on
        num_episodes: Number of episodes to run
        max_steps: Steps per episode
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating PPO model: {model_path}")
    print(f"Load pattern: {load_pattern}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Load model
    env = create_env(load_pattern)
    model = PPO.load(model_path)
    
    all_metrics = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        episode_metrics = {
            'rewards': [],
            'instances': [],
            'load': [],
            'queue_size': [],
            'influx': [],
            'actions': [],
            'costs': []
        }
        
        for step in range(max_steps):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Collect metrics
            episode_metrics['rewards'].append(reward)
            episode_metrics['instances'].append(len(env.instances))
            episode_metrics['load'].append(env.load)
            episode_metrics['queue_size'].append(env.queue_size)
            episode_metrics['influx'].append(env.influx)
            episode_metrics['actions'].append(env.actions[action])
            episode_metrics['costs'].append(env.total_cost)
            
            if done or truncated:
                break
        
        all_metrics.append(episode_metrics)
        
        print(f"  Episode {episode + 1:2d}: "
              f"Reward={sum(episode_metrics['rewards']):7.2f}, "
              f"Cost=${env.total_cost:8.2f}, "
              f"Avg Load={np.mean(episode_metrics['load']):5.1f}%, "
              f"Avg Queue={np.mean(episode_metrics['queue_size']):6.2f}")
    
    # Aggregate metrics
    aggregated = {
        'avg_reward': np.mean([sum(m['rewards']) for m in all_metrics]),
        'std_reward': np.std([sum(m['rewards']) for m in all_metrics]),
        'avg_cost': np.mean([m['costs'][-1] for m in all_metrics]),
        'std_cost': np.std([m['costs'][-1] for m in all_metrics]),
        'avg_load': np.mean([np.mean(m['load']) for m in all_metrics]),
        'avg_queue': np.mean([np.mean(m['queue_size']) for m in all_metrics]),
        'avg_instances': np.mean([np.mean(m['instances']) for m in all_metrics]),
        'episodes': all_metrics
    }
    
    print(f"\n{'='*70}")
    print("Evaluation Summary:")
    print(f"  Average Reward: {aggregated['avg_reward']:.2f} ± {aggregated['std_reward']:.2f}")
    print(f"  Average Cost: ${aggregated['avg_cost']:.2f} ± ${aggregated['std_cost']:.2f}")
    print(f"  Average Load: {aggregated['avg_load']:.1f}%")
    print(f"  Average Queue: {aggregated['avg_queue']:.2f}")
    print(f"  Average Instances: {aggregated['avg_instances']:.1f}")
    print(f"{'='*70}\n")
    
    env.close()
    
    return aggregated


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent with GPU acceleration')
    parser.add_argument('--pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'SINUSOIDAL', 'STEADY', 'SPIKE', 'POISSON', 'RANDOM'],
                       help='Load pattern for training')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps (default: 100,000)')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for saving')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Train with GPU
    model, model_path = train_ppo_gpu(
        load_pattern=args.pattern,
        total_timesteps=args.timesteps,
        model_name=args.name
    )
    
    # Evaluate if requested
    if args.eval:
        metrics = evaluate_ppo(model_path, load_pattern=args.pattern, num_episodes=10)
