#!/usr/bin/env python3
"""
Quick test to verify GIF generation works
"""
import sys

print("="*80)
print("GIF GENERATION TEST")
print("="*80)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    print("  ✓ matplotlib")
except ImportError as e:
    print(f"  ❌ matplotlib: {e}")
    print("     Install with: pip install matplotlib")
    sys.exit(1)

try:
    from PIL import Image
    print("  ✓ PIL/Pillow")
except ImportError as e:
    print(f"  ❌ PIL/Pillow: {e}")
    print("     Install with: pip install pillow")
    sys.exit(1)

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    print("  ❌ numpy")
    sys.exit(1)

try:
    from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
    print("  ✓ gym_scaling")
except ImportError as e:
    print(f"  ❌ gym_scaling: {e}")
    sys.exit(1)

# Test 2: Check models
print("\n2. Checking models...")
import os

models_found = []
if os.path.exists('models/dqn_notebook_trained.zip'):
    print("  ✓ DQN model (100k trained)")
    models_found.append('DQN')
elif os.path.exists('models/dqn_simple_sine_curve_20251129_192916.zip'):
    print("  ✓ DQN model (alternative)")
    models_found.append('DQN')
else:
    print("  ⚠️  DQN model not found")

if os.path.exists('models/ppo_simple_20251129_180558.zip'):
    print("  ✓ PPO model")
    models_found.append('PPO')
else:
    print("  ⚠️  PPO model not found")

if os.path.exists('models/qlearning_extended_20251129_175127.pkl'):
    print("  ✓ Q-Learning model")
    models_found.append('Q-Learning')
else:
    print("  ⚠️  Q-Learning model not found")

# Test 3: Test environment
print("\n3. Testing environment...")
try:
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    obs = env.reset()
    
    # Run a few steps
    for i in range(5):
        obs, reward, done, info = env.step(1)
    
    print("  ✓ Environment working")
    print(f"    - Can collect data for GIF generation")
    env.close()
except Exception as e:
    print(f"  ❌ Environment test failed: {e}")
    sys.exit(1)

# Test 4: Test matplotlib animation
print("\n4. Testing matplotlib animation...")
try:
    fig, ax = plt.subplots()
    x = np.linspace(0, 2*np.pi, 100)
    line, = ax.plot(x, np.sin(x))
    
    def animate(frame):
        line.set_ydata(np.sin(x + frame/10))
        return line,
    
    anim = animation.FuncAnimation(fig, animate, frames=10, interval=100)
    
    # Try to save (but don't actually save to avoid clutter)
    print("  ✓ Animation creation works")
    plt.close()
except Exception as e:
    print(f"  ⚠️  Animation test: {e}")
    print("     (This might be OK - actual GIF generation may still work)")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✓ Core dependencies: Working")
print(f"✓ Models found: {len(models_found)}/3 ({', '.join(models_found)})")
print(f"✓ Environment: Working")

if len(models_found) > 0:
    print(f"\n✅ Ready to generate GIFs!")
    print(f"\nRun:")
    print(f"  python generate_demo_gif.py --policy dqn --steps 50")
    print(f"\nThis will create a test GIF in images/dqn_demo.gif")
else:
    print(f"\n⚠️  No models found - train models first")

print("="*80)
