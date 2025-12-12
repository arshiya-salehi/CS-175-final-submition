"""
Load generator functions for different traffic patterns
"""
import math
import random
import numpy as np


def steady_load(step, max_influx, offset):
    """Generate constant steady load"""
    return int(offset + (max_influx - offset) * 0.5)


def sinusoidal_load(step, max_influx, offset, frequency=0.01):
    """Generate sinusoidal variation in load"""
    return int(math.ceil((np.sin(float(step) * frequency) + 1) * (max_influx - offset) / 2) + offset)


def spike_load(step, max_influx, offset, spike_frequency=500, spike_magnitude=0.8):
    """
    Generate load with abrupt spikes
    spike_frequency: how often spikes occur (in steps)
    spike_magnitude: how high spikes go (0-1, relative to max_influx)
    """
    base_load = offset + (max_influx - offset) * 0.3
    
    # Check if we're in a spike period
    if step % spike_frequency < 20:  # Spike lasts 20 steps
        spike_load = offset + (max_influx - offset) * spike_magnitude
        # Gradual spike up and down
        if step % spike_frequency < 10:
            # Increasing
            progress = (step % spike_frequency) / 10.0
            load = base_load + (spike_load - base_load) * progress
        else:
            # Decreasing
            progress = ((step % spike_frequency) - 10) / 10.0
            load = spike_load - (spike_load - base_load) * progress
        return int(load)
    else:
        return int(base_load)


def poisson_load(step, max_influx, offset, lambda_param=None):
    """
    Generate Poisson-arrival workload (stochastic)
    lambda_param: average arrival rate (if None, uses mid-point between offset and max)
    """
    if lambda_param is None:
        lambda_param = offset + (max_influx - offset) * 0.5
    
    # Generate Poisson-distributed load
    # Use exponential distribution to model inter-arrival times
    load = np.random.poisson(lambda_param)
    # Clamp to valid range
    load = max(offset, min(max_influx, load))
    return int(load)


def random_load(step, max_influx, offset):
    """Generate completely random load"""
    return random.randint(int(offset), int(max_influx))


# Dictionary mapping pattern names to functions
LOAD_PATTERNS = {
    'STEADY': {
        'function': steady_load,
        'options': {}
    },
    'SINUSOIDAL': {
        'function': sinusoidal_load,
        'options': {'frequency': 0.01}
    },
    'SPIKE': {
        'function': spike_load,
        'options': {'spike_frequency': 500, 'spike_magnitude': 0.8}
    },
    'POISSON': {
        'function': poisson_load,
        'options': {'lambda_param': None}
    },
    'RANDOM': {
        'function': random_load,
        'options': {}
    }
}

