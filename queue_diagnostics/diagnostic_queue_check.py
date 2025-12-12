"""
Diagnostic script to check if queue is actually being tracked
"""
import numpy as np
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS

# Create environment
env = ScalingEnv()
env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
env.change_rate = 1

obs = env.reset()
print("Initial state:")
print(f"  Queue size: {env.queue_size}")
print(f"  Instances: {len(env.instances)}")
print(f"  Load: {env.load}")
print(f"  Influx: {env.influx}")

# Run a few steps with NO SCALING (action=1 = do nothing)
print("\nRunning 20 steps with NO SCALING (to build up queue):")
for i in range(20):
    obs, reward, done, info = env.step(1)  # Do nothing
    if i % 5 == 0:
        print(f"  Step {i:2d}: Queue={env.queue_size:6.1f}, Influx={env.influx:4d}, "
              f"Instances={len(env.instances):2d}, Load={env.load:3.0f}%")

print(f"\nFinal queue size: {env.queue_size}")
print(f"Max queue in history: {max(env.hi_queue_size) if env.hi_queue_size else 0}")

# Now try with aggressive scaling DOWN
print("\n\nRunning 20 steps with AGGRESSIVE SCALE DOWN:")
env.reset()
for i in range(20):
    obs, reward, done, info = env.step(0)  # Scale down
    if i % 5 == 0:
        print(f"  Step {i:2d}: Queue={env.queue_size:6.1f}, Influx={env.influx:4d}, "
              f"Instances={len(env.instances):2d}, Load={env.load:3.0f}%")

print(f"\nFinal queue size: {env.queue_size}")
print(f"Max queue in history: {max(env.hi_queue_size) if env.hi_queue_size else 0}")
