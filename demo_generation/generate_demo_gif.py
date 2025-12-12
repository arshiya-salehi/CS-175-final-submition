#!/usr/bin/env python3
"""
Generate animated GIF demonstrations of autoscaling policies
Similar to sin_input_result.gif
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pickle
import argparse
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


def run_episode(policy_name, model=None, q_table=None, max_steps=200, load_pattern='SINE_CURVE'):
    """Run an episode and collect data for visualization"""
    print(f"Running {policy_name}...")
    
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS[load_pattern]
    env.change_rate = 1
    obs = env.reset()
    
    data = {
        'steps': [],
        'influx': [],
        'instances': [],
        'queue': [],
        'load': [],
        'actions': [],
        'reward': [],
        'cost': []
    }
    
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
        
        # Record data
        data['steps'].append(step)
        data['influx'].append(env.influx)
        data['instances'].append(len(env.instances))
        data['queue'].append(env.queue_size)
        data['load'].append(env.load)
        data['actions'].append(env.actions[action])
        data['reward'].append(total_reward)
        data['cost'].append(env.total_cost)
        
        if done:
            break
    
    env.close()
    print(f"  Completed {len(data['steps'])} steps")
    return data


def create_animated_gif(data, policy_name, output_file, fps=10):
    """Create an animated GIF showing the autoscaling in action"""
    print(f"Creating GIF for {policy_name}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{policy_name} Autoscaling Policy', fontsize=16, fontweight='bold')
    
    # Prepare data
    steps = data['steps']
    max_step = max(steps)
    
    def animate(frame):
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Get data up to current frame
        current_steps = steps[:frame+1]
        current_influx = data['influx'][:frame+1]
        current_instances = data['instances'][:frame+1]
        current_queue = data['queue'][:frame+1]
        current_load = data['load'][:frame+1]
        
        # Plot 1: Influx and Instances
        ax1 = axes[0]
        ax1.plot(current_steps, current_influx, 'b-', linewidth=2, label='Incoming Load')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(current_steps, current_instances, 'g-', linewidth=2, label='Instances')
        
        ax1.set_ylabel('Incoming Load', color='b', fontsize=12, fontweight='bold')
        ax1_twin.set_ylabel('Number of Instances', color='g', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, max_step)
        ax1.set_ylim(0, max(data['influx']) * 1.1)
        ax1_twin.set_ylim(0, max(data['instances']) * 1.2)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Queue Size
        ax2 = axes[1]
        ax2.fill_between(current_steps, current_queue, alpha=0.3, color='red')
        ax2.plot(current_steps, current_queue, 'r-', linewidth=2, label='Queue Size')
        ax2.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Warning')
        ax2.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Critical')
        ax2.set_ylabel('Queue Size', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, max_step)
        ax2.set_ylim(0, max(max(data['queue']), 100) * 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Plot 3: Load and Cost
        ax3 = axes[2]
        ax3.plot(current_steps, current_load, 'purple', linewidth=2, label='CPU Load %')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='High Load')
        ax3.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Low Load')
        ax3.set_ylabel('CPU Load (%)', color='purple', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, max_step)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        # Add statistics text
        if frame < len(data['steps']):
            stats_text = (
                f"Step: {data['steps'][frame]}\n"
                f"Instances: {data['instances'][frame]}\n"
                f"Load: {data['load'][frame]:.0f}%\n"
                f"Queue: {data['queue'][frame]:.0f}\n"
                f"Cost: ${data['cost'][frame]:.2f}\n"
                f"Reward: {data['reward'][frame]:.2f}"
            )
            fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Create animation
    frames = len(steps)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps, repeat=True)
    
    # Save as GIF
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"  ✓ Saved {output_file}")


def create_comparison_gif(all_data, output_file, fps=10):
    """Create a side-by-side comparison GIF of all policies"""
    print("Creating comparison GIF...")
    
    num_policies = len(all_data)
    fig, axes = plt.subplots(num_policies, 3, figsize=(18, 5*num_policies))
    fig.suptitle('Autoscaling Policy Comparison', fontsize=18, fontweight='bold')
    
    if num_policies == 1:
        axes = axes.reshape(1, -1)
    
    # Get max steps across all policies
    max_steps = max(max(d['steps']) for d in all_data.values())
    
    def animate(frame):
        for idx, (policy_name, data) in enumerate(all_data.items()):
            # Clear axes
            for ax in axes[idx]:
                ax.clear()
            
            # Get data up to current frame (or max available)
            current_frame = min(frame, len(data['steps']) - 1)
            current_steps = data['steps'][:current_frame+1]
            
            # Plot 1: Influx and Instances
            ax1 = axes[idx, 0]
            ax1.plot(current_steps, data['influx'][:current_frame+1], 'b-', linewidth=2)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(current_steps, data['instances'][:current_frame+1], 'g-', linewidth=2)
            ax1.set_ylabel('Load', color='b', fontsize=10)
            ax1_twin.set_ylabel('Instances', color='g', fontsize=10)
            ax1.set_title(f'{policy_name}', fontsize=12, fontweight='bold')
            ax1.set_xlim(0, max_steps)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Queue
            ax2 = axes[idx, 1]
            ax2.fill_between(current_steps, data['queue'][:current_frame+1], alpha=0.3, color='red')
            ax2.plot(current_steps, data['queue'][:current_frame+1], 'r-', linewidth=2)
            ax2.set_ylabel('Queue', fontsize=10)
            ax2.set_xlim(0, max_steps)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Load
            ax3 = axes[idx, 2]
            ax3.plot(current_steps, data['load'][:current_frame+1], 'purple', linewidth=2)
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=40, color='orange', linestyle='--', alpha=0.5)
            ax3.set_ylabel('CPU %', fontsize=10)
            ax3.set_xlim(0, max_steps)
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            if idx == num_policies - 1:
                ax1.set_xlabel('Time Step', fontsize=10)
                ax2.set_xlabel('Time Step', fontsize=10)
                ax3.set_xlabel('Time Step', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Create animation
    frames = max_steps
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps, repeat=True)
    
    # Save as GIF
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer='pillow', fps=fps)
    plt.close()
    
    print(f"  ✓ Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate animated GIF demos')
    parser.add_argument('--policy', type=str, default='all',
                       choices=['all', 'threshold', 'qlearning', 'dqn', 'ppo'])
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of steps to simulate')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIF')
    parser.add_argument('--output-dir', type=str, default='images',
                       help='Output directory for GIFs')
    parser.add_argument('--load-pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'RANDOM'])
    parser.add_argument('--comparison', action='store_true',
                       help='Create side-by-side comparison GIF')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANIMATED GIF GENERATOR")
    print("="*80)
    print(f"Configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  FPS: {args.fps}")
    print(f"  Output: {args.output_dir}/")
    print(f"  Load Pattern: {args.load_pattern}")
    print("="*80)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = {}
    
    # Load and run models
    if args.policy in ['all', 'threshold']:
        data = run_episode('Threshold', max_steps=args.steps, load_pattern=args.load_pattern)
        all_data['Threshold'] = data
        if not args.comparison:
            create_animated_gif(data, 'Threshold', 
                              f'{args.output_dir}/threshold_demo.gif', fps=args.fps)
    
    if args.policy in ['all', 'qlearning']:
        try:
            with open('models/qlearning_extended_20251129_175127.pkl', 'rb') as f:
                q_table = pickle.load(f)
            data = run_episode('Q-Learning', q_table=q_table, 
                             max_steps=args.steps, load_pattern=args.load_pattern)
            all_data['Q-Learning'] = data
            if not args.comparison:
                create_animated_gif(data, 'Q-Learning',
                                  f'{args.output_dir}/qlearning_demo.gif', fps=args.fps)
        except Exception as e:
            print(f"❌ Q-Learning not available: {e}")
    
    if args.policy in ['all', 'dqn']:
        try:
            dqn_model = DQN.load('models/dqn_notebook_trained.zip')
            data = run_episode('DQN', model=dqn_model,
                             max_steps=args.steps, load_pattern=args.load_pattern)
            all_data['DQN'] = data
            if not args.comparison:
                create_animated_gif(data, 'DQN',
                                  f'{args.output_dir}/dqn_demo.gif', fps=args.fps)
        except:
            try:
                dqn_model = DQN.load('models/dqn_simple_sine_curve_20251129_192916.zip')
                data = run_episode('DQN', model=dqn_model,
                                 max_steps=args.steps, load_pattern=args.load_pattern)
                all_data['DQN'] = data
                if not args.comparison:
                    create_animated_gif(data, 'DQN',
                                      f'{args.output_dir}/dqn_demo.gif', fps=args.fps)
            except Exception as e:
                print(f"❌ DQN not available: {e}")
    
    if args.policy in ['all', 'ppo']:
        try:
            ppo_model = PPO.load('models/ppo_simple_20251129_180558.zip')
            data = run_episode('PPO', model=ppo_model,
                             max_steps=args.steps, load_pattern=args.load_pattern)
            all_data['PPO'] = data
            if not args.comparison:
                create_animated_gif(data, 'PPO',
                                  f'{args.output_dir}/ppo_demo.gif', fps=args.fps)
        except Exception as e:
            print(f"❌ PPO not available: {e}")
    
    # Create comparison GIF if requested
    if args.comparison and len(all_data) > 1:
        create_comparison_gif(all_data, f'{args.output_dir}/comparison_demo.gif', fps=args.fps)
    
    print("\n" + "="*80)
    print("✓ DONE!")
    print(f"GIFs saved in {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
