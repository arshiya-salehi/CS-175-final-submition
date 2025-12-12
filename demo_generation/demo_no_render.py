#!/usr/bin/env python3
"""
Demo without rendering - works on all systems
Shows text-based progress and final comparison
"""
import numpy as np
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


def run_demo(policy_name, model=None, q_table=None, max_steps=200, 
             load_pattern='SINE_CURVE', verbose=True):
    """Run a single policy demo without rendering"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {policy_name}")
        print(f"{'='*80}\n")
    
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS[load_pattern]
    env.change_rate = 1
    obs = env.reset()
    
    total_reward = 0
    max_queue = 0
    queue_history = []
    instance_history = []
    load_history = []
    
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
        max_queue = max(max_queue, env.queue_size)
        
        # Track history
        queue_history.append(env.queue_size)
        instance_history.append(len(env.instances))
        load_history.append(env.load)
        
        # Print progress
        if verbose and step % 20 == 0:
            print(f"Step {step:3d}: Instances={len(env.instances):2d}, "
                  f"Load={env.load:3.0f}%, Queue={env.queue_size:6.1f}, "
                  f"Cost=${env.total_cost:8.2f}, Reward={total_reward:7.2f}")
        
        if done:
            if verbose:
                print(f"\nEpisode ended at step {step}")
            break
    
    # Calculate statistics
    avg_queue = np.mean(queue_history)
    avg_instances = np.mean(instance_history)
    avg_load = np.mean(load_history)
    
    if verbose:
        print(f"\n{'-'*80}")
        print(f"Final Statistics for {policy_name}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Total Cost: ${env.total_cost:.2f}")
        print(f"  Max Queue: {max_queue:.1f}")
        print(f"  Avg Queue: {avg_queue:.2f}")
        print(f"  Avg Instances: {avg_instances:.1f}")
        print(f"  Avg Load: {avg_load:.1f}%")
        print(f"{'-'*80}\n")
    
    env.close()
    
    return {
        'policy': policy_name,
        'reward': total_reward,
        'cost': env.total_cost,
        'max_queue': max_queue,
        'avg_queue': avg_queue,
        'avg_instances': avg_instances,
        'avg_load': avg_load
    }


def print_ascii_chart(results):
    """Print a simple ASCII bar chart"""
    print("\n" + "="*80)
    print("VISUAL COMPARISON")
    print("="*80)
    
    # Normalize rewards for visualization (0-100 scale)
    rewards = [r['reward'] for r in results]
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    print("\nReward Comparison (higher is better):")
    for result in results:
        normalized = int(((result['reward'] - min_reward) / (max_reward - min_reward + 0.001)) * 50)
        bar = '█' * normalized
        print(f"{result['policy']:<15} {bar} {result['reward']:.2f}")
    
    # Cost comparison
    costs = [r['cost'] for r in results]
    max_cost = max(costs)
    
    print("\nCost Comparison (lower is better):")
    for result in results:
        normalized = int((result['cost'] / max_cost) * 50)
        bar = '█' * normalized
        print(f"{result['policy']:<15} {bar} ${result['cost']:.2f}")
    
    # Queue comparison
    print("\nMax Queue Comparison (lower is better):")
    max_queues = [r['max_queue'] for r in results]
    max_q = max(max_queues) if max(max_queues) > 0 else 1
    
    for result in results:
        normalized = int((result['max_queue'] / max_q) * 50)
        bar = '█' * normalized if normalized > 0 else ''
        print(f"{result['policy']:<15} {bar} {result['max_queue']:.1f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Demo without rendering')
    parser.add_argument('--policy', type=str, default='all',
                       choices=['all', 'threshold', 'qlearning', 'dqn', 'ppo'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--load-pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'RANDOM'])
    parser.add_argument('--quiet', action='store_true',
                       help='Only show final results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AUTOSCALING POLICY DEMO (No Rendering)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  Load Pattern: {args.load_pattern}")
    print(f"  Verbose: {not args.quiet}")
    print("="*80)
    
    results = []
    models = {}
    
    # Load models
    if args.policy in ['all', 'threshold']:
        print("\n✓ Threshold policy ready")
        models['Threshold'] = {'type': 'threshold'}
    
    if args.policy in ['all', 'qlearning']:
        try:
            with open('models/qlearning_extended_20251129_175127.pkl', 'rb') as f:
                q_table = pickle.load(f)
            print(f"✓ Q-Learning loaded ({len(q_table)} states)")
            models['Q-Learning'] = {'type': 'qlearning', 'q_table': q_table}
        except Exception as e:
            print(f"❌ Q-Learning not found: {e}")
    
    if args.policy in ['all', 'dqn']:
        try:
            dqn_model = DQN.load('models/dqn_notebook_trained.zip')
            print(f"✓ DQN loaded ({dqn_model.num_timesteps:,} timesteps)")
            models['DQN'] = {'type': 'dqn', 'model': dqn_model}
        except Exception as e:
            try:
                dqn_model = DQN.load('models/dqn_simple_sine_curve_20251129_192916.zip')
                print(f"✓ DQN loaded ({dqn_model.num_timesteps:,} timesteps)")
                models['DQN'] = {'type': 'dqn', 'model': dqn_model}
            except:
                print(f"❌ DQN not found")
    
    if args.policy in ['all', 'ppo']:
        try:
            ppo_model = PPO.load('models/ppo_simple_20251129_180558.zip')
            print(f"✓ PPO loaded ({ppo_model.num_timesteps:,} timesteps)")
            models['PPO'] = {'type': 'ppo', 'model': ppo_model}
        except Exception as e:
            print(f"❌ PPO not found: {e}")
    
    if not models:
        print("\n❌ No models available!")
        return
    
    # Run demos
    verbose = not args.quiet
    
    for policy_name, policy_data in models.items():
        if policy_data['type'] == 'threshold':
            result = run_demo(policy_name, max_steps=args.steps,
                            load_pattern=args.load_pattern, verbose=verbose)
        elif policy_data['type'] == 'qlearning':
            result = run_demo(policy_name, q_table=policy_data['q_table'],
                            max_steps=args.steps, load_pattern=args.load_pattern,
                            verbose=verbose)
        else:
            result = run_demo(policy_name, model=policy_data['model'],
                            max_steps=args.steps, load_pattern=args.load_pattern,
                            verbose=verbose)
        
        results.append(result)
    
    # Print ASCII charts
    if len(results) > 1:
        print_ascii_chart(results)
    
    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Policy':<15} {'Reward':<12} {'Cost':<15} {'Max Queue':<12} {'Avg Queue':<12}")
    print("-"*80)
    for result in results:
        print(f"{result['policy']:<15} {result['reward']:>11.2f} "
              f"${result['cost']:>13.2f} {result['max_queue']:>11.1f} "
              f"{result['avg_queue']:>11.2f}")
    print("="*80)
    
    # Winner
    best_reward = max(results, key=lambda x: x['reward'])
    lowest_cost = min(results, key=lambda x: x['cost'])
    lowest_queue = min(results, key=lambda x: x['max_queue'])
    
    print("\n🏆 Winners:")
    print(f"  Best Reward: {best_reward['policy']} ({best_reward['reward']:.2f})")
    print(f"  Lowest Cost: {lowest_cost['policy']} (${lowest_cost['cost']:.2f})")
    print(f"  Lowest Queue: {lowest_queue['policy']} ({lowest_queue['max_queue']:.1f})")
    print()


if __name__ == '__main__':
    main()
