# Generalist Multirotor Control

**Embodiment-conditioned Generalist Control for Multirotor Aerial Robots**  
Orestis Konstantaropoulos, Welf Rehberg, Mihir Kulkarni, Kostas Alexis

---

## Overview

This repository provides a simulation and RL training framework for training a single generalist controller that can fly a diverse set of multirotor airframes without airframe-specific re-tuning. The default configuration targets 6-motor platforms, but quadrotors are supported — see [Changing motor count](#changing-motor-count) below.

Key components:

- **Simulator** (`simulator/`): GPU-parallelised rigid-body multirotor dynamics built on [NVIDIA Warp](https://github.com/NVIDIA/warp). Supports up to 2048+ parallel environments. `MultirotorDynamicsEnv` is the base class; `CustomObsEnv` extends it with configurable observation spaces, domain randomisation, trajectory tracking (Langevin / Lissajous), and importance sampling.
- **Airframe generation** (`airframe_generation/`): Procedural sampler for generating diverse multirotor morphologies. Pre-generated datasets (`valid_airframe_config_6_*.pkl`) cover symmetric, planar, and random 6-motor layouts. Other motor counts (e.g. quadrotors) are supported by changing `num_motors` in `airframe_generation/sampler.py` and the corresponding warp kernel shape in `simulator/multirotor_dynamics_warp_kernels.py`.
- **RL training** (`rl_training/rl_games/`): PPO training via a modified [rl-games](https://github.com/Denys88/rl_games) backend. Feedforward (FFN) and recurrent (RNN) policy variants are supported. Configuration lives in `ppo_aerial_quad.yaml`. Real-world-tested sim2real checkpoint is provided under `rl_training/rl_games/networks/sim2real/`.
- **Evaluation** (`scripts/train_evaluate.py`): End-to-end script that optionally trains a policy and then rolls it out, computing hover and trajectory tracking metrics (position RMSE, velocity RMSE, angular velocity RMSE) with per-embodiment breakdowns and multi-panel plots.

### Observation scenarios

The observation space is controlled by the `scenario` flag, which determines what robot descriptor information is provided to the policy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ntnu-arl/generalist_multirotor_control.git
cd generalist_multirotor_control
```

### 2. Create a conda environment

```bash
conda create -n gmc python=3.8 -y
conda activate gmc
```

### 3. Install PyTorch

Tested with PyTorch 2.4.1 and CUDA 12.1:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install the modified RL-Games package

```bash
pip install -e libs/rl_games
```

### 6. Install this package

```bash
pip install -e .
```

---

## Training and Evaluation

The main entry point is `scripts/train_evaluate.py`.

**Train and evaluate scenario 8 with an RNN policy:**

```bash
python scripts/train_evaluate.py --scenario 8 --rnn --tr
```

**Evaluate a pre-trained checkpoint (no training):**

```bash
python scripts/train_evaluate.py --scenario 8 --rnn
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `-1` | Observation scenario (see table above) |
| `--rnn` | `False` | Use recurrent (LSTM) policy instead of FFN |
| `--tr` | `False` | Run training before evaluation |
| `--robot_num` | `-1` | Fix evaluation to a single robot index |
| `--seed` | `23` | Random seed |
| `--dr` | `False` | Enable descriptor (domain) randomisation |

Evaluation outputs (plots + error tensors) are saved to `scenario_<N>/`.


## Changing motor count

The framework is not limited to 6-motor platforms. To use a different number of motors (e.g. quadrotors):

1. **`airframe_generation/sampler.py`** — set `num_motors` when generating or loading an airframe config dataset.
2. **`simulator/multirotor_dynamics_warp_kernels.py`** — the warp kernel matrix shapes are parameterised by `num_motors`; no hardcoded sizes need to change as long as the config passed to `CustomObsEnv` carries the correct `num_motors` value.

Everything downstream (observation dimension, action dimension, allocation matrix shape) is derived automatically from `num_motors` at initialisation time.

---

## Citation

```bibtex
@article{konstantaropoulos2026generalist, title={Embodiment-conditioned Generalist Control for Multirotor Aerial Robots}, author={Konstantaropoulos, Orestis and Rehberg, Welf and Kulkarni, Mihir and Alexis, Kostas}, journal={arXiv preprint arXiv:2606.10857}, year={2026} }
```
