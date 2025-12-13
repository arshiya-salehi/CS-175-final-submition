# 🚀 Training Guide - Smart AutoScaler RL Agent

A simple guide to train reinforcement learning models for cloud autoscaling.

## 📋 Prerequisites

### 1. Install Python Dependencies
```bash
# Install external dependencies
pip install -r requirements.txt

# Install the gym_scaling package in editable mode (recommended)
pip install -e .
```

**Note**: If you're running scripts from the project root directory, the `pip install -e .` step is optional since Python will find the `gym_scaling` folder automatically. However, it's recommended for a proper setup.

### 2. Key Dependencies
- **Python**: 3.8-3.11
- **PyTorch**: 2.3.0+ (with MPS support for M4 Mac)
- **Stable-Baselines3**: 2.7.0
- **Gymnasium**: 1.2.0+
- **NumPy**: 2.3.5

### 3. Hardware Requirements
- **Minimum**: 4-core CPU, 8GB RAM
- **Recommended**: M4 MacBook Air (GPU acceleration)
- **Storage**: 2GB free space

## 🎯 Quick Start (5 minutes)

### Test Everything Works
```bash
# Quick test with 1,000 timesteps
python src/train_dqn_simple.py --timesteps 1000 --pattern SINE_CURVE

python src/train_qlearning_extended.py --episodes 1000

python src/train_ppo_m4_gpu.py --timesteps 1000 --pattern SINE_CURVE

# Verify the model was created
ls -la models/dqn_simple_sine_curve_*.zip
```

## 🏋️ Full Training (30-60 minutes)

### 1. Train Q-Learning (5-10 minutes)
```bash
python src/train_qlearning_extended.py --episodes 10000 --eval
```
**Output**: `models/qlearning_extended_YYYYMMDD_HHMMSS.pkl`

### 2. Train DQN (15-20 minutes)
```bash
python src/train_dqn_simple.py --timesteps 100000 --pattern SINE_CURVE
```
**Output**: `models/dqn_simple_sine_curve_YYYYMMDD_HHMMSS.zip`

### 3. Train PPO (20-25 minutes)
```bash
python src/train_ppo_m4_gpu.py --timesteps 100000 --eval
```
**Output**: `models/ppo_simple_YYYYMMDD_HHMMSS.zip`

## 🎮 Test Your Models

### Verify All Models Work
```bash
python src/verify_all_models.py
```

### See Models in Action
```bash
python src/demo_all_models.py
```

## 📊 Training Options

### Load Patterns
- `SINE_CURVE` - Predictable daily cycles (recommended)
- `RANDOM` - Random load spikes
- `STEADY` - Constant load

### Example Commands
```bash
# Different patterns
python src/train_dqn_simple.py --pattern RANDOM --timesteps 50000
python src/train_dqn_simple.py --pattern STEADY --timesteps 50000

# Longer training
python src/train_dqn_simple.py --timesteps 200000 --pattern SINE_CURVE
```

## ⚡ Performance Tips

### M4 MacBook Air Users
- GPU acceleration is automatic (MPS)
- 3-4x faster than CPU training
- Check GPU usage: `python -c "import torch; print('MPS:', torch.backends.mps.is_available())"`

### Training Times (M4 Mac)
- **Q-Learning**: 5-10 minutes (10k episodes)
- **DQN**: 15-20 minutes (100k timesteps)  
- **PPO**: 20-25 minutes (100k timesteps)

## 🔧 Troubleshooting

### Common Issues

**1. Dependencies Missing**
```bash
# Install/update external packages
pip install --upgrade torch stable-baselines3 gymnasium numpy

# If imports fail, install the local package
pip install -e .
```

**2. GPU Not Working**
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**3. Training Too Slow**
- Use fewer timesteps for testing: `--timesteps 10000`
- GPU scripts are faster: `train_dqn_m4_gpu.py` vs `train_dqn_simple.py`

**4. Out of Memory**
- Training automatically falls back to CPU if GPU runs out of memory
- Reduce timesteps if needed

## 📁 Output Files

After training, you'll have:
```
models/
├── qlearning_extended_YYYYMMDD_HHMMSS.pkl    # Q-Learning model
├── dqn_simple_sine_curve_YYYYMMDD_HHMMSS.zip # DQN model  
└── ppo_simple_YYYYMMDD_HHMMSS.zip            # PPO model
```

## 🎯 Expected Results

### Performance (lower is better)
- **Random Policy**: -500 to -800
- **Q-Learning**: -100 to -300  
- **DQN**: -50 to -200
- **PPO**: -30 to -150 (best)

### What Success Looks Like
- ✅ All training scripts complete without errors
- ✅ Models are saved to `models/` directory
- ✅ `verify_all_models.py` shows all tests pass
- ✅ `demo_all_models.py` shows intelligent autoscaling behavior

## 🚨 Need Help?

### Quick Diagnostics
```bash
# Check Python version
python --version

# Check key packages
python -c "import torch, stable_baselines3, gymnasium; print('All packages OK')"

# Check GPU
python -c "import torch; print('Device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"

# Test environment (this will fail if you forgot pip install -e .)
python -c "from gym_scaling.envs.scaling_env import ScalingEnv; print('Environment OK')"
```

### Still Having Issues?
1. **Make sure you're in the project root directory** (where setup.py is located)
2. **If you get import errors**, run `pip install -e .`
3. Try the quick test first: `python src/train_dqn_simple.py --timesteps 1000`
4. Check the detailed `src/README.md` for advanced options

---

**Ready to train? Start with the Quick Start section above! 🚀**