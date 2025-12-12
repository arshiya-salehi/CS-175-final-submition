#!/usr/bin/env python3
"""
Visual Demo: Compare All Autoscaling Policies
Shows Q-Learning, DQN, PPO, and Threshold policies in action
"""
import numpy as np
import pickle
import time
import argparse
from stable_baselines3 import DQN, PPO
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


def discretize_state(obs, bins=10):
    """Discretize state for Q-Learning"""
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
    """Threshold-based policy (like Kubernetes HPA)"""
    load = obs[1] * 100
    queue_size = obs[4]
    if load > high_threshold or queue_size > 100:
        return 2  # Add instance
    elif load < low_threshold and queue_size == 0:
        return 0  # Remove instance
    else:
        return 1  # Do nothing


def run_demo(policy_name, model=None, q_table=None, max_steps=200, 
             render=True, delay=0.05, load_pattern='SINE_CURVE'):
    """Run a single policy demo"""
    print(f"\n{'='*80}")
    print(f"Running: {policy_name}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS[load_pattern]
    env.change_rate = 1  # Change load every step for challenging demo
    obs = env.reset()
    
    total_reward = 0
    max_queue = 0
    
    for step in range(max_steps):
        # Select action based on policy
        if policy_name == 'Threshold':
            action = threshold_policy(obs)
        elif policy_name == 'Q-Learning':
            state = discretize_state(obs)
            action = np.argmax(q_table[state]) if state in q_table else 1
        else:  # DQN or PPO
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        
        # Take action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        max_queue = max(max_queue, env.queue_size)
        
        # Render
        if render:
            try:
                env.render()
                time.sleep(delay)
            except Exception as e:
                if step == 0:
                    print(f"\n⚠️  Rendering failed: {e}")
                    print("    Continuing without visualization...")
                    render = False  # Disable rendering for rest of episode
        
        # Print stats every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d}: Instances={len(env.instances):2d}, "
                  f"Load={env.load:3.0f}%, Queue={env.queue_size:6.1f}, "
                  f"Cost=${env.total_cost:8.2f}, Reward={total_reward:7.2f}")
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Final stats
    print(f"\n{'-'*80}")
    print(f"Final Statistics for {policy_name}:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Cost: ${env.total_cost:.2f}")
    print(f"  Max Queue: {max_queue:.1f}")
    print(f"  Final Instances: {len(env.instances)}")
    print(f"  Avg Load: {np.mean(list(env.hi_load)):.1f}%")
    print(f"{'-'*80}\n")
    
    env.close()
    return {
        'policy': policy_name,
        'reward': total_reward,
        'cost': env.total_cost,
        'max_queue': max_queue
    }


def main():
    parser = argparse.ArgumentParser(description='Demo all autoscaling policies')
    parser.add_argument('--policy', type=str, default='all',
                       choices=['all', 'threshold', 'qlearning', 'dqn', 'ppo'],
                       help='Which policy to demo (default: all)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of steps to run (default: 200)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visual rendering')
    parser.add_argument('--delay', type=float, default=0.05,
                       help='Delay between steps in seconds (default: 0.05)')
    parser.add_argument('--load-pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'RANDOM'],
                       help='Load pattern to use (default: SINE_CURVE)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AUTOSCALING POLICY DEMO")
    print("="*80)
    print(f"Configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  Render: {not args.no_render}")
    print(f"  Load Pattern: {args.load_pattern}")
    print(f"  Delay: {args.delay}s")
    print("="*80)
    
    results = []
    
    # Load models
    models = {}
    
    # Threshold (always available)
    if args.policy in ['all', 'threshold']:
        print("\n✓ Threshold policy ready")
        models['Threshold'] = {'type': 'threshold'}
    
    # Q-Learning
    if args.policy in ['all', 'qlearning']:
        try:
            with open('models/qlearning_extended_20251129_175127.pkl', 'rb') as f:
                q_table = pickle.load(f)
            print(f"✓ Q-Learning loaded ({len(q_table)} states)")
            models['Q-Learning'] = {'type': 'qlearning', 'q_table': q_table}
        except Exception as e:
            print(f"❌ Q-Learning not found: {e}")
    
    # DQN
    if args.policy in ['all', 'dqn']:
        try:
            # Try the 100k trained model first
            dqn_model = DQN.load('models/dqn_notebook_trained.zip')
            print(f"✓ DQN loaded ({dqn_model.num_timesteps:,} timesteps)")
            models['DQN'] = {'type': 'dqn', 'model': dqn_model}
        except Exception as e:
            try:
                # Fallback to other model
                dqn_model = DQN.load('models/dqn_simple_sine_curve_20251129_192916.zip')
                print(f"✓ DQN loaded ({dqn_model.num_timesteps:,} timesteps)")
                models['DQN'] = {'type': 'dqn', 'model': dqn_model}
            except Exception as e2:
                print(f"❌ DQN not found: {e2}")
    
    # PPO
    if args.policy in ['all', 'ppo']:
        try:
            ppo_model = PPO.load('models/ppo_simple_20251129_180558.zip')
            print(f"✓ PPO loaded ({ppo_model.num_timesteps:,} timesteps)")
            models['PPO'] = {'type': 'ppo', 'model': ppo_model}
        except Exception as e:
            print(f"❌ PPO not found: {e}")
    
    if not models:
        print("\n❌ No models available to demo!")
        return
    
    # Run demos
    for policy_name, policy_data in models.items():
        if policy_data['type'] == 'threshold':
            result = run_demo(
                policy_name, 
                max_steps=args.steps,
                render=not args.no_render,
                delay=args.delay,
                load_pattern=args.load_pattern
            )
        elif policy_data['type'] == 'qlearning':
            result = run_demo(
                policy_name,
                q_table=policy_data['q_table'],
                max_steps=args.steps,
                render=not args.no_render,
                delay=args.delay,
                load_pattern=args.load_pattern
            )
        else:  # DQN or PPO
            result = run_demo(
                policy_name,
                model=policy_data['model'],
                max_steps=args.steps,
                render=not args.no_render,
                delay=args.delay,
                load_pattern=args.load_pattern
            )
        
        results.append(result)
        
        if len(models) > 1:
            input("\nPress Enter to continue to next policy...")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Policy':<15} {'Total Reward':<15} {'Total Cost':<15} {'Max Queue':<12}")
    print("-"*80)
    for result in results:
        print(f"{result['policy']:<15} {result['reward']:>14.2f} "
              f"${result['cost']:>13.2f} {result['max_queue']:>11.1f}")
    print("="*80)


if __name__ == '__main__':
    main()
