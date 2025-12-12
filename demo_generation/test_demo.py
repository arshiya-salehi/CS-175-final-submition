#!/usr/bin/env python3
"""
Quick test to verify demo system is working
"""
import sys
import os

print("="*80)
print("DEMO SYSTEM TEST")
print("="*80)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("  ✓ numpy")
except ImportError as e:
    print(f"  ❌ numpy: {e}")
    sys.exit(1)

try:
    from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
    print("  ✓ gym_scaling")
except ImportError as e:
    print(f"  ❌ gym_scaling: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import DQN, PPO
    print("  ✓ stable_baselines3")
except ImportError as e:
    print(f"  ❌ stable_baselines3: {e}")
    sys.exit(1)

try:
    import pyglet
    print("  ✓ pyglet (rendering)")
except ImportError as e:
    print(f"  ⚠️  pyglet: {e} (rendering will not work)")

# Test 2: Check models
print("\n2. Checking models...")
models_found = []

if os.path.exists('models/qlearning_extended_20251129_175127.pkl'):
    print("  ✓ Q-Learning model found")
    models_found.append('Q-Learning')
else:
    print("  ⚠️  Q-Learning model not found")

if os.path.exists('models/dqn_simple_sine_curve_20251129_192916.zip'):
    print("  ✓ DQN model found")
    models_found.append('DQN')
else:
    print("  ⚠️  DQN model not found")

if os.path.exists('models/ppo_simple_20251129_180558.zip'):
    print("  ✓ PPO model found")
    models_found.append('PPO')
else:
    print("  ⚠️  PPO model not found")

# Test 3: Test environment
print("\n3. Testing environment...")
try:
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    obs = env.reset()
    print(f"  ✓ Environment created")
    print(f"    - Observation shape: {obs.shape}")
    print(f"    - Initial instances: {len(env.instances)}")
    print(f"    - Initial queue: {env.queue_size}")
    
    # Take a few steps
    for i in range(5):
        obs, reward, done, info = env.step(1)
    print(f"  ✓ Environment step working")
    print(f"    - Queue after 5 steps: {env.queue_size}")
    
    env.close()
except Exception as e:
    print(f"  ❌ Environment test failed: {e}")
    sys.exit(1)

# Test 4: Test rendering (optional)
print("\n4. Testing rendering...")
try:
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    obs = env.reset()
    
    # Try to render (might fail if no display)
    try:
        env.render()
        print("  ✓ Rendering works!")
        env.close()
    except Exception as e:
        print(f"  ⚠️  Rendering not available: {e}")
        print("     (This is OK - you can use --no-render)")
        env.close()
except Exception as e:
    print(f"  ⚠️  Rendering test skipped: {e}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✓ Core system: Working")
print(f"✓ Models found: {len(models_found)}/3 ({', '.join(models_found)})")

if len(models_found) == 0:
    print("\n⚠️  WARNING: No models found!")
    print("   Train your models first or check the models/ directory")
else:
    print(f"\n✅ Ready to demo! You can run:")
    print(f"   python demo_all_models.py")
    if len(models_found) < 3:
        print(f"\n   Note: Only {len(models_found)} model(s) available")
        print(f"   Available: {', '.join(models_found)}")

print("="*80)
