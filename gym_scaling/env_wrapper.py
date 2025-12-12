"""
Environment wrapper to fix observation space shape for stable-baselines3
"""
import gymnasium as gym
from gym import spaces
import numpy as np


class ScalingEnvWrapper(gym.Wrapper):
    """
    Wrapper to fix observation space shape mismatch
    The original environment defines observation_space as (1, 5) but returns (5,)
    This wrapper fixes it to (5,) to match the actual observation
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Fix observation space to match actual observation shape
        original_shape = self.observation_space.shape
        if len(original_shape) == 2 and original_shape[0] == 1:
            # Convert (1, 5) to (5,)
            self.observation_space = spaces.Box(
                low=self.observation_space.low.flatten(),
                high=self.observation_space.high.flatten(),
                shape=(original_shape[1],),
                dtype=self.observation_space.dtype
            )
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Handle both tuple (obs, info) and single obs return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        # Ensure observation is flattened if needed
        if obs.ndim > 1:
            obs = obs.flatten()
        return obs, info
    
    def step(self, action):
        result = self.env.step(action)
        # Handle both old (4 values) and new (5 values) gym API
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        # Ensure observation is flattened if needed
        if obs.ndim > 1:
            obs = obs.flatten()
        return obs, reward, terminated, truncated, info

