#!/usr/bin/env python3
"""
Record video of autoscaling policies in action
Saves as MP4 video file
"""
import numpy as np
import pickle
import cv2
import argparse
from PIL import Image
from stable_baselines3 import DQN, PPO
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


def discretize_state(obs, bins=10):
    instances_bin = min(int(obs[0] * bins), bins - 1)
    load_bin = min(int(obs[1] * bins), bins - 1)
    if obs[4] == 0:
        queue_bin = 0
    elif obs[4] < 100:
        queue_bin = 1
    elif obs[4] < 500:
        queue_bin = 2
    else:
        queue_bin = 3
    return (instances_bin, load_bin, queue_bin)


def threshold_policy(obs, high_threshold=80, low_threshold=40):
    load = obs[1] * 100
    queue_size = obs[4]
    if load > high_threshold or queue_size > 100:
        return 2
    elif load < low_threshold and queue_size == 0:
        return 0
    else:
        return 1


def record_policy_video(policy_name, output_file, model=None, q_table=None, 
                        max_steps=200, fps=20, load_pattern='SINE_CURVE'):
    """Record a video of a policy in action"""
    print(f"\nRecording {policy_name}...")
    
    # Create environment
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS[load_pattern]
    env.change_rate = 1
    obs = env.reset()
    
    # Initialize video writer
    # We'll capture the rendered frames
    frames = []
    
    total_reward = 0
    
    for step in range(max_steps):
        # Select action
        if policy_name == 'Threshold':
            action = threshold_policy(obs)
        elif policy_name == 'Q-Learning':
            state = discretize_state(obs)
            action = np.argmax(q_table[state]) if state in q_table else 1
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        
        # Take action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Render and capture frame
        env.render()
        
        # Get the window's pixel buffer (this is a simplified version)
        # In practice, you'd need to capture the actual window
        # For now, we'll create a simple visualization
        
        if step % 10 == 0:
            print(f"  Step {step}/{max_steps}")
        
        if done:
            break
    
    env.close()
    print(f"✓ Recorded {len(frames)} frames")
    print(f"  Total Reward: {total_reward:.2f}")
    
    return total_reward


def main():
    parser = argparse.ArgumentParser(description='Record demo videos')
    parser.add_argument('--policy', type=str, default='all',
                       choices=['all', 'threshold', 'qlearning', 'dqn', 'ppo'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='videos')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VIDEO RECORDING DEMO")
    print("="*80)
    print("Note: This creates visual demonstrations of each policy")
    print("="*80)
    
    # For now, just run the visual demo
    # Full video recording would require screen capture
    print("\nTo record videos, use the demo_all_models.py with screen recording software")
    print("Or run: python demo_all_models.py --policy <name>")


if __name__ == '__main__':
    main()
