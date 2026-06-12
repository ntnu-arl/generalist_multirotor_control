import os
import shutil
import sys
import yaml
import torch
import random
import pickle as pkl
import numpy as np
import subprocess
from argparse import ArgumentParser

# Gym & RL-Games
import gym
from gym import spaces
import rl_games.common.env_configurations as env_configurations
from rl_games.common import vecenv
from rl_games.torch_runner import Runner

# Custom project imports
from generalist_multirotor_control.simulator.custom_obs_env import CustomObsEnv

from generalist_multirotor_control.rl_training.rl_games.runner import ExtractObsWrapper

# Network builders
from generalist_multirotor_control.rl_training.rl_games.networks.gen_policy_network import (
    RnnTestNetBuilder, DecTestNetBuilder
)
from rl_games.algos_torch import model_builder

# Register custom networks
model_builder.register_network('rnntestnet', RnnTestNetBuilder)
model_builder.register_network('dectestnet', DecTestNetBuilder)


# Define the path to the weights and config files

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_num",
        type=int,
        default=-1,
        help="Robot number to use. -1 for random.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=-1,
        help="Scenario to use. -1 for random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--rnn",
        default=False,
        action='store_true',
        help="Whether to use RNN policy network.",
    )
    parser.add_argument(
        "--dr",
        default=False,
        action='store_true',
        help="Whether to use descriptor randomization.",
    )
    parser.add_argument(
        "--im",
        default=False,
        action='store_true',
        help="Whether to use imitation learning.",
    )
    parser.add_argument(
        "--tr",
        action='store_true',
        help="Run training before evaluation.",
    )

    args = parser.parse_args()
    return vars(args)

def load_config(config_path):
    # yaml safe load
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Convert the config to a dictionary if it's not already
    if isinstance(config, dict):
        config["params"]["config"]["player"] = {"use_vecenv": True}
        print("Config loaded successfully.")
        return config
    else:
        raise ValueError("Configuration file must contain a dictionary.")


def inspect_restored_weights(model, weights_path, map_location):
    """Print whether the restored model tensors match the checkpoint tensors."""
    checkpoint = torch.load(weights_path, map_location=map_location)

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint_state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint_state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        checkpoint_state = checkpoint
    else:
        print(f"[LOAD CHECK] Unsupported checkpoint format: {type(checkpoint)}")
        return

    checkpoint_state = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint_state.items()
    }
    model_state = model.state_dict()

    matched_keys = []
    mismatched_keys = []
    shape_mismatch_keys = []

    for key, checkpoint_value in checkpoint_state.items():
        if key not in model_state:
            continue

        model_value = model_state[key]
        matched_keys.append(key)

        if not torch.is_tensor(model_value) or not torch.is_tensor(checkpoint_value):
            if model_value != checkpoint_value:
                mismatched_keys.append(key)
            continue

        if model_value.shape != checkpoint_value.shape:
            shape_mismatch_keys.append(key)
            continue

        checkpoint_value = checkpoint_value.detach().to(
            device=model_value.device,
            dtype=model_value.dtype,
        )
        if not torch.equal(model_value.detach(), checkpoint_value):
            mismatched_keys.append(key)

    missing_model_keys = [key for key in model_state.keys() if key not in checkpoint_state]
    unexpected_checkpoint_keys = [key for key in checkpoint_state.keys() if key not in model_state]

    print(f"[LOAD CHECK] checkpoint: {weights_path}")
    print(f"[LOAD CHECK] model tensors: {len(model_state)}")
    print(f"[LOAD CHECK] matched checkpoint keys: {len(matched_keys)}")
    print(f"[LOAD CHECK] exact tensor matches: {len(matched_keys) - len(mismatched_keys) - len(shape_mismatch_keys)}")
    print(f"[LOAD CHECK] shape mismatches: {len(shape_mismatch_keys)}")
    print(f"[LOAD CHECK] missing model keys: {len(missing_model_keys)}")
    print(f"[LOAD CHECK] unexpected checkpoint keys: {len(unexpected_checkpoint_keys)}")

    if mismatched_keys:
        print(
            f"[LOAD CHECK] first value mismatch: {mismatched_keys[0]}"
        )
    if shape_mismatch_keys:
        print(
            f"[LOAD CHECK] first shape mismatch: {shape_mismatch_keys[0]}"
        )
    if missing_model_keys:
        print(
            f"[LOAD CHECK] first missing model key: {missing_model_keys[0]}"
        )
    if unexpected_checkpoint_keys:
        print(
            f"[LOAD CHECK] first unexpected checkpoint key: {unexpected_checkpoint_keys[0]}"
        )

##################################################################
# ENVIRONMENT SETUP
##################################################################

def create_env(config):
    env = CustomObsEnv(
        config=config,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    env = ExtractObsWrapper(env)
    return env



class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.action_space_dim),
            np.ones(self.env.action_space_dim),
            # -np.ones(4),
            # np.ones(4),
        )
        info["observation_space"] = spaces.Box(
            -np.Inf* np.ones(self.env.observation_space_dim),
            np.Inf * np.ones(self.env.observation_space_dim),
        )
        print(info["action_space"], info["observation_space"])
        return info


env_configurations.register(
    "position_setpoint_task_custom",
    {
        "env_creator": lambda **kwargs: CustomObsEnv(config=robot_configs, device="cuda:0"),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)



# ============================================================================ #
#                            TRAIN                                             #
# ============================================================================ #


args = parse_args()
sce = args['scenario']
seed = args['seed']
seed = 1234567892 
if not args['rnn']:
    base_cmd = ["python3", "runner.py", "--task=position_setpoint_task_custom", "--num_envs=2048", f"--experiment={sce}ffn", f"--scenario={sce}", f"--seed", f"{seed}", f"--importance_sampling"] #, f"--randomize_descriptor" if args['dr'] else ""]  f"--track"] 
    train_cmds = [base_cmd]
else:
    base_cmd = ["python3", "runner.py", "--task=position_setpoint_task_custom", "--num_envs=2048", f"--experiment={sce}rnn", f"--scenario={sce}", f"--seed", f"{seed}", "--rnn", f"--randomize_descriptor" if args['dr'] else ""] #, f"--importance_sampling"]
    train_cmds = [base_cmd]
IMITATION = args['im']  # Whether to use imitation learning
to_train = args['tr']  # Whether to run training jobs first (controlled by --tr flag)

if to_train:
    print("=" * 80)
    print("Stage 1: Running training jobs (runner.py)")
    print("=" * 80)

    for cmd in train_cmds:
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd="../rl_training/rl_games")
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}: {' '.join(cmd)}")
            sys.exit(result.returncode)

    print("=" * 80)
    print("Stage 1 complete")
    print("=" * 80)


import generalist_multirotor_control as mds


package_prefix = os.path.dirname(mds.__file__) + "/"
runs_dir = package_prefix + "rl_training/rl_games/runs/"
run_dirs = os.listdir(runs_dir)
setting = f"{args['scenario']}{'rnn' if args['rnn'] else 'ffn'}"
print("Looking for setting: ", setting)
matching_dirs = []
for d in run_dirs:
    s = d.split('_')
    if IMITATION and 'im' in s:
        matching_dirs.append(d)
    elif not IMITATION and setting in s[0] and ('l' not in s):
        matching_dirs.append(d)
if not matching_dirs:
    raise RuntimeError(f"No run directory found for setting '{setting}' in {runs_dir}")
# Pick the most recently modified directory (i.e. the one just trained)
run_path = max(matching_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
print("Using run: ", run_path)
if not IMITATION:
    weights_path = runs_dir + run_path + "/nn/" + setting + ".pth"
    # file_names = os.listdir(runs_dir + run_path + "/nn/")
    # last_model = [fn for fn in file_names if "last_" in fn][0]
    # weights_path = runs_dir + run_path + "/nn/" + last_model
else:
    weights_path = runs_dir + run_path + "/nn/im.pth"
config_path = package_prefix + "rl_training/rl_games/ppo_aerial_quad.yaml"



filename = package_prefix + "/airframe_generation/valid_airframe_config_6_all.pkl"


with open(filename, "rb") as f:
    robot_configs = pkl.load(f)
robot_configs['scenario'] = args['scenario'] # -1 for random
robot_configs['one_point'] = True # initialize at one point for easier evaluation and visualization
robot_configs['robot_num'] = args['robot_num'] # -1 for random
robot_configs['seed'] = args['seed']  # set seed for evaluation environment
robot_configs['repeat_configs'] = 10 #00  # repeat each config this many times to fill up NUM_ENVS
robot_configs['max_episode_steps'] =600*2
robot_configs['dt'] = 0.01 
robot_configs['lissajous'] = False 
robot_configs['randomize_descriptor'] = None #True if args['dr'] else None
# Note: Environment will be created by RL-Games when creating the player


### -----------------


### -----------------



PLOT = True
from matplotlib import pyplot as plt
def plot_rewards(action_reward, pos_reward, vel_reward, angvel_reward, ang_acc_reward = None, proximity_reward = None, proximity = None, output_dir="."):
    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(action_reward, label='Action Reward', linewidth=2)
    plt.plot(pos_reward, label='Position Reward', linewidth=2)
    plt.plot(vel_reward, label='Velocity Reward', linewidth=2)
    plt.plot(angvel_reward, label='Angular Velocity Reward', linewidth=2)
    plt.plot(ang_acc_reward, label='Angular Acceleration Reward', linewidth=2)
    if proximity_reward is not None:
        plt.plot(proximity_reward, label='Proximity Reward', linewidth=2)
    if proximity is not None:
        plt.plot(proximity, label='Proximity', linewidth=2)

    # Formatting
    plt.title('Reward Components Over Time', fontsize=18)
    plt.xlabel('Timestep', fontsize=16)
    plt.ylabel('Reward Value', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save and show
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_reward_components_plot.png', dpi=300)
    plt.show(block=False)

import numpy as np

def compute_hover_ss_metrics(
    state_np,
    target_pos=np.zeros(3),
    target_vel=np.zeros(3),
    target_angvel=np.zeros(3),
    ss_frac=0.3,
    num_spawns=10,
):
    """
    state_np: [T, num_samples, 13]
      state = [pos(3), quat(4), vel(3), angvel(3)]

    num_samples = num_embodiments * num_spawns
    """

    T, num_samples, D = state_np.shape
    assert D == 13
    assert num_samples % num_spawns == 0

    num_embodiments = num_samples // num_spawns
    ss_start = int((1.0 - ss_frac) * T)

    ss = state_np[ss_start:]  # [T_ss, num_samples, 13]

    pos = ss[:, :, 0:3]
    vel = ss[:, :, 7:10]
    angvel = ss[:, :, 10:13]

    target_pos = np.asarray(target_pos).reshape(1, 1, 3)
    target_vel = np.asarray(target_vel).reshape(1, 1, 3)
    target_angvel = np.asarray(target_angvel).reshape(1, 1, 3)

    # Per rollout steady-state RMSE: shape [num_samples]
    pos_ss_err = np.sqrt(np.mean(np.sum((pos - target_pos) ** 2, axis=-1), axis=0))
    vel_ss_err = np.sqrt(np.mean(np.sum((vel - target_vel) ** 2, axis=-1), axis=0))
    angvel_ss_err = np.sqrt(np.mean(np.sum((angvel - target_angvel) ** 2, axis=-1), axis=0))

    def summarize(errors):
        return {
            "mean": errors.mean(),
            "std": errors.std(ddof=1),
            "median": np.median(errors),
            "p90": np.percentile(errors, 90),
            "p95": np.percentile(errors, 95),
            "min": errors.min(),
            "max": errors.max(),
        }

    metrics = {
        "pos": summarize(pos_ss_err),
        "vel": summarize(vel_ss_err),
        "angvel": summarize(angvel_ss_err),
    }

    # Optional: reshape to [num_embodiments, num_spawns]
    per_emb = {
        "pos": pos_ss_err.reshape(num_embodiments, num_spawns),
        "vel": vel_ss_err.reshape(num_embodiments, num_spawns),
        "angvel": angvel_ss_err.reshape(num_embodiments, num_spawns),
    }

    return metrics, per_emb


def _summarize_errors(errors):
    std_val = errors.std(ddof=1) if errors.size > 1 else 0.0
    return {
        "mean": float(errors.mean()),
        "std": float(std_val),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "min": float(errors.min()),
        "max": float(errors.max()),
    }


def compute_tracking_ss_metrics(
    state_np,
    target_pos_np,
    target_vel_np=None,
    target_angvel=np.zeros(3),
    ss_frac=0.3,
    num_spawns=10,
):
    """Steady-state metrics against time-varying target position/velocity trajectories.

    state_np: [T, num_samples, 13]
    target_pos_np: [T, num_samples, 3]
    target_vel_np: [T, num_samples, 3] (optional)
    """

    T_state, N_state, D = state_np.shape
    if D != 13:
        raise ValueError(f"Expected state dim 13, got {D}")

    if target_pos_np.ndim != 3 or target_pos_np.shape[2] != 3:
        raise ValueError(f"target_pos_np must be [T, N, 3], got shape {target_pos_np.shape}")

    T = min(T_state, target_pos_np.shape[0])
    N = min(N_state, target_pos_np.shape[1])
    if T <= 1 or N <= 0:
        raise ValueError(f"Invalid overlap between state and target: T={T}, N={N}")

    if target_vel_np is None:
        target_vel_np = np.zeros((T, N, 3), dtype=state_np.dtype)
    else:
        if target_vel_np.ndim != 3 or target_vel_np.shape[2] != 3:
            raise ValueError(f"target_vel_np must be [T, N, 3], got shape {target_vel_np.shape}")
        target_vel_np = target_vel_np[:T, :N]

    state = state_np[:T, :N]
    target_pos = target_pos_np[:T, :N]
    target_angvel = np.asarray(target_angvel).reshape(1, 1, 3)

    ss_start = int((1.0 - ss_frac) * T)
    ss_state = state[ss_start:]
    ss_target_pos = target_pos[ss_start:]
    ss_target_vel = target_vel_np[ss_start:]

    pos = ss_state[:, :, 0:3]
    vel = ss_state[:, :, 7:10]
    angvel = ss_state[:, :, 10:13]

    pos_ss_err = np.sqrt(np.mean(np.sum((pos - ss_target_pos) ** 2, axis=-1), axis=0))
    vel_ss_err = np.sqrt(np.mean(np.sum((vel - ss_target_vel) ** 2, axis=-1), axis=0))
    angvel_ss_err = np.sqrt(np.mean(np.sum((angvel - target_angvel) ** 2, axis=-1), axis=0))

    metrics = {
        "pos": _summarize_errors(pos_ss_err),
        "vel": _summarize_errors(vel_ss_err),
        "angvel": _summarize_errors(angvel_ss_err),
    }

    if N % num_spawns != 0:
        print(f"[METRICS] N={N} not divisible by num_spawns={num_spawns}. Using num_spawns=1 for per_emb.")
        num_spawns = 1

    num_embodiments = N // num_spawns
    per_emb = {
        "pos": pos_ss_err.reshape(num_embodiments, num_spawns),
        "vel": vel_ss_err.reshape(num_embodiments, num_spawns),
        "angvel": angvel_ss_err.reshape(num_embodiments, num_spawns),
    }

    return metrics, per_emb


def get_lissajous_targets_for_mask(eval_env, keep_mask, max_steps):
    """Return lissajous target pos/vel as [T, N_selected, 3], aligned with keep_mask."""
    base_env = eval_env
    if not hasattr(base_env, "lissajous_traj") and hasattr(base_env, "env"):
        base_env = base_env.env

    # if not hasattr(base_env, "lissajous_traj"):
    #     return None, None

    traj = base_env.lissajous_traj
    # if not isinstance(traj, dict) or "pos" not in traj:
    #     return None, None

    pos = traj["pos"]  # [T, num_envs, 3] or [T, 3]
    vel = traj.get("vel", None)
    print('bika kai')
    print(vel.shape, pos.shape)

    selected_idx = torch.nonzero(keep_mask, as_tuple=True)[0]
    print(selected_idx.shape, keep_mask.shape, eval_env.num_envs)

    if pos.shape[0] == eval_env.num_envs:
        sel_pos = pos[selected_idx]
        sel_vel = vel[selected_idx] if isinstance(vel, torch.Tensor) else None
    elif pos.shape[0] == selected_idx.numel():
        sel_pos = pos
        sel_vel = vel if isinstance(vel, torch.Tensor) else None
    else:
        print(
            f"[METRICS] Unable to align lissajous_traj with selected envs: "
            f"traj_batch={pos.shape[0]}, selected={selected_idx.numel()}, num_envs={eval_env.num_envs}"
        )
        return None, None

    # Convert from [N, T, 3] to [T, N, 3]
    target_pos_np = sel_pos.detach().cpu().numpy().transpose(1, 0, 2)
    target_vel_np = None
    if sel_vel is not None:
        target_vel_np = sel_vel.detach().cpu().numpy().transpose(1, 0, 2)
    return target_pos_np, target_vel_np



def main():
    global ROBOT_TO_PICK

    # Create output directory for plots
    output_dir = f"scenario_{args['scenario']}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}/")

    # Set seeds for reproducibility
    import random
    import numpy as np
    seed = args['seed'] 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make sure CuDNN is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # num envs in config:
    config = load_config(config_path)
    # We'll set num_envs after creating the player since we need the environment instance
    if args["rnn"]:
        print('Using RNN policy network')
        config["params"]["network"]['name'] = "rnntestnet"
    else:
        print('Using Feedforward policy network')
        config["params"]["network"]['name'] = "dectestnet"
    runner = Runner()
    runner.load(config)
    print("Config loaded into runner.", config)
    # Create player and load checkpoint
    player = runner.create_player()
    
    # Now we can access the environment through the player
    env = player.env.env
    
    # Set the correct number of environments in config (for consistency)
    config["params"]["config"]["env_config"]["num_envs"] = env.num_envs
    config["params"]["config"]["num_actors"] = env.num_envs
    
    if IMITATION:
         student_obs_shape = 49 # obs shape for student model trained with imitation learning (only state, no descriptor)
    player.restore(weights_path)
    inspect_restored_weights(player.model, weights_path, env.device)
    player.reset()

    # print the model architecture
    print(player.model)
    # Reset the environment
    obs = env.reset()
    obs = obs['obs']
    num_samples_h = 20
    state_tensor_list_all = []
    big_state_tensor_list_all = []
    actions_all = []
    total_dones = torch.zeros(env.num_envs, dtype=torch.bool, device="cuda:0")
    reward_stats = 0
    action_reward, pos_reward, vel_reward, angvel_reward, ang_acc_reward, proximity_reward, proximity = [], [], [], [], [], [], []
    

    crashes, crashes_angvel, crashes_pos, crashes_vel, crashes_timeout = 0, 0, 0, 0, 0
    crashes_no_ped_idx = torch.zeros(env.num_envs, device=env.device)
    torch.set_printoptions(precision=10, sci_mode=False)

    for step in range(env.max_episode_steps):
        if IMITATION:
            obs = obs[:,:student_obs_shape]
        action = player.get_action(obs, is_deterministic=True)
        obs, reward, done, info = env.step(action)
        obs = obs['obs']
        reward_stats += (reward.mean().item())
        total_dones += done

        crashes_tensor = (info['crash_dict']['angvel'] + info['crash_dict']['pos'] + info['crash_dict']['vel'] + info['crash_dict']['timeout'])
        crashes += crashes_tensor.sum()
        crashes_no_ped_idx += crashes_tensor
        crashes_angvel += (info['crash_dict']['angvel']).sum()
        crashes_pos += (info['crash_dict']['pos']).sum()
        crashes_vel += (info['crash_dict']['vel']).sum()
        crashes_timeout += (info['crash_dict']['timeout']).sum()
        action_reward.append(info['crash_dict']['action_dif'].mean().item())
        pos_reward.append(info['crash_dict']['pos_rew'].mean().item())
        vel_reward.append(info['crash_dict']['vel_rew'].mean().item())
        angvel_reward.append(info['crash_dict']['angvel_rew'].mean().item())
        ang_acc_reward.append(info['crash_dict']['ang_acc'].mean().item())
        state_tensor_list_all.append(env.root_state_tensor.clone())
        big_state_tensor_list_all.append(obs.clone())
        actions_all.append(action.clone())

    # reset statistics
    spawn_count = int(robot_configs.get('repeat_configs', 10))
    print(total_dones.sum(), (total_dones > 2).sum(), reward_stats)
    print("crashes (angular velocity):", crashes_angvel.item())
    print("crashes (position):", crashes_pos.item())
    print("crashes (velocity):", crashes_vel.item())
    print("crashes (timeout):", crashes_timeout.item())
    crashes_no_ped_idx = crashes_no_ped_idx.reshape(-1, spawn_count).sum(dim=1)
    torch.set_printoptions(threshold=10_000)
    print("crashes per robot:", torch.topk(crashes_no_ped_idx, k = 10))
    no_crashed_configs = (crashes_no_ped_idx < 1) #CHANGE BACK
    print('no_crashed_configs:', no_crashed_configs.sum().item(), '/', no_crashed_configs.shape[0])
    no_crashed_configs_emb_level = no_crashed_configs.clone()  # [num_embodiments] boolean, before repeat
    no_crashed_configs = no_crashed_configs.repeat_interleave(spawn_count)
    

    state_tensor_list_all_arr = torch.stack(state_tensor_list_all, dim=0)[:, no_crashed_configs, :]
    big_state_tensor_list_all_arr = torch.stack(big_state_tensor_list_all, dim=0)[:, no_crashed_configs, :]
    actions_all = torch.stack(actions_all, dim=0)[:, no_crashed_configs, :]

    # Targets used for downstream metrics/plots.
    if robot_configs.get('lissajous', False):
        try:
            target_pos_np_liss, lissajous_target_vel_np = get_lissajous_targets_for_mask(env, no_crashed_configs, robot_configs['max_episode_steps'])
            target_pos_list_all_arr = torch.from_numpy(target_pos_np_liss).to(state_tensor_list_all_arr.device)
        except Exception as err:
            print(f"[WARN] Failed to fetch lissajous targets ({err}). Falling back to zero hover targets.")
            target_pos_list_all_arr = torch.zeros_like(state_tensor_list_all_arr[:, :, 0:3])
            lissajous_target_vel_np = None
    else:
        target_pos_list_all_arr = torch.zeros_like(state_tensor_list_all_arr[:, :, 0:3])
        lissajous_target_vel_np = None

    # Observations already contain tracking errors:
    #   pos error [0:3], vel error [7:10], angvel error [10:13].
    pos_error_vec = big_state_tensor_list_all_arr[:, :, 0:3]
    vel_error_vec = big_state_tensor_list_all_arr[:, :, 7:10]
    angvel_error_vec = big_state_tensor_list_all_arr[:, :, 10:13]

    m_pos_error = pos_error_vec.norm(dim=-1)
    m_vel_error = vel_error_vec.norm(dim=-1)
    m_angvel_error = angvel_error_vec.norm(dim=-1)

    # Average component-wise errors over repeats: [T, N_survived_embs, spawn_count, 3] -> [T, N_survived_embs, 3]
    n_survived = no_crashed_configs_emb_level.sum().item()
    pos_error_avg   = pos_error_vec.reshape(-1, n_survived, spawn_count, 3).mean(dim=2)    # [T, N_survived, 3]
    vel_error_avg   = vel_error_vec.reshape(-1, n_survived, spawn_count, 3).mean(dim=2)    # [T, N_survived, 3]
    angvel_error_avg = angvel_error_vec.reshape(-1, n_survived, spawn_count, 3).mean(dim=2) # [T, N_survived, 3]

    error_filename = "lissajous_tracking_errors.pt" if robot_configs.get('lissajous', False) else "hover_tracking_errors.pt"
    error_save_path = os.path.join(output_dir, error_filename)
    torch.save(
        {
            "position_error": pos_error_avg.cpu(),       # [T, N_survived, 3] component-wise
            "velocity_error": vel_error_avg.cpu(),       # [T, N_survived, 3] component-wise
            "angular_velocity_error": angvel_error_avg.cpu(),  # [T, N_survived, 3] component-wise
            "no_crashed_configs": no_crashed_configs_emb_level.cpu(),  # [num_embodiments] bool
            "dt": robot_configs['dt'],
            "scenario": args['scenario'],
            "rnn": args['rnn'],
            "lissajous": bool(robot_configs.get('lissajous', False)),
            "spawn_count": spawn_count,
        },
        error_save_path,
    )
    print(f"Saved tracking error tensors to: {error_save_path}")

    pos_error_per_axis = pos_error_vec.abs()
    mean_pos_error_per_robot = m_pos_error[-500:,:].mean(dim=0).reshape(-1,10).mean(dim=1)
    print("best robots:", torch.topk(mean_pos_error_per_robot, k = 10, largest = False), mean_pos_error_per_robot.shape)
    print("Reward: {}, percentage of crashes: {}".format(reward_stats, crashes.item()/(env.num_envs+crashes.item())))
    #print('TOTAL POSITION ERROR TRAJ: ', m_pos_error.mean(), '+-', m_pos_error.std())
    print('TOTAL POSITION ERROR TRAJ: ', m_pos_error.mean(), '+-', m_pos_error.reshape(-1,10).std(axis=1).mean())
    #print(f"FINAL POSITION ERROR: {m_pos_error[-1,:].mean():.4f} ± {m_pos_error[-1,:].std():.4f} m")
    print(f"FINAL POSITION ERROR: {m_pos_error[-1,:].mean():.4f} ± {m_pos_error[-1,:].reshape(-1,10).std(axis=1).mean():.4f} m")
    print(f"FINAL POSITION ERROR PER AXIS X: {pos_error_per_axis[-1,:, 0].mean():.4f} ± {pos_error_per_axis[-1,:, 0].reshape(-1,10).std(axis=1).mean():.4f} m")
    print(f"FINAL POSITION ERROR PER AXIS Y: {pos_error_per_axis[-1,:, 1].mean():.4f} ± {pos_error_per_axis[-1,:, 1].reshape(-1,10).std(axis=1).mean():.4f} m")
    print(f"FINAL POSITION ERROR PER AXIS Z: {pos_error_per_axis[-1,:, 2].mean():.4f} ± {pos_error_per_axis[-1,:, 2].reshape(-1,10).std(axis=1).mean():.4f} m")
    plot_rewards(action_reward, pos_reward, vel_reward, angvel_reward, ang_acc_reward, proximity_reward, proximity, output_dir)


    if PLOT:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        # Enhanced plotting for lissajous trajectory comparison
        state_np = state_tensor_list_all_arr.cpu().numpy()
        big_state_np = big_state_tensor_list_all_arr.cpu().numpy()
        target_pos_np = target_pos_list_all_arr.cpu().numpy()
        no_crashed_c = no_crashed_configs.cpu().numpy()
        time_np = np.arange(len(state_np)) * robot_configs['dt']
        action_np = (actions_all.cpu().numpy() + 1)* 0.4*9.81*0.5 * 0.5
        data = {"state_np": state_np, "actions": action_np, 'no_crashed_configs': no_crashed_c, 'big_state_np': big_state_np,}
        angvel_norm = np.linalg.norm(state_np[:, :, 10:13], axis=2)  # [time_steps, num_samples]
        if robot_configs.get('lissajous', False):
            metrics, per_emb = compute_tracking_ss_metrics(
                state_np,
                target_pos_np,
                target_vel_np=lissajous_target_vel_np,
                ss_frac=0.3,
                num_spawns=robot_configs['repeat_configs'],
            )
        else:
            metrics, per_emb = compute_hover_ss_metrics(
                state_np,
                target_pos=np.array([0.0, 0.0, 0.0]),
                ss_frac=0.3,
                num_spawns=robot_configs['repeat_configs'],
            )

        print(metrics)
       
        if True:
            # Compute statistics across multiple robots
            # state_np shape: [time_steps, num_samples, state_dim]
            # target_pos_np shape: [time_steps, num_samples, 3]
            
            # Calculate mean and standard deviation across robots
            state_mean = np.mean(state_np, axis=1)  # [time_steps, state_dim]
            state_std = np.std(state_np, axis=1)    # [time_steps, state_dim]

            
            target_mean = np.mean(target_pos_np, axis=1)  # [time_steps, 3]
            target_std = np.std(target_pos_np, axis=1)    # [time_steps, 3]
            
            # Colors for robot trajectories
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            
            # Calculate position error statistics 
            pos_errors = np.linalg.norm(target_pos_np - state_np[:, :, 0:3], axis=2)  # [time_steps, num_samples]
            pos_error_mean = np.mean(pos_errors, axis=1)
            pos_error_std = np.mean(np.std(pos_errors.reshape(-1,10), axis=1))
            
            p = np.random.randint(0,1000)  # Random starting point for robot selection
            
            # === FIGURE 1: POSITION ANALYSIS ===
            fig_pos = plt.figure(figsize=(18, 12))
            fig_pos.suptitle(f'Position Analysis - Scenario {args["scenario"]} {"(RNN)" if args["rnn"] else "(FFN)"}', fontsize=20, fontweight='bold')
            
            # 2D Trajectory comparison (X-Y plane) with variance
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(target_mean[:, 0], target_mean[:, 1], 'r-', 
                     label='Target Hover Trajectory', linewidth=3, alpha=0.9)
            ax1.plot(state_mean[:, 0], state_mean[:, 1], 'b-', 
                     label=f'Mean Robot Path (n={num_samples_h})', linewidth=2, alpha=0.9)
            
            # Add variance ellipses
            theta = np.linspace(0, 2*np.pi, 100)
            for i in range(0, len(time_np), max(1, len(time_np)//20)):
                if state_std[i, 0] > 0.001 or state_std[i, 1] > 0.001:
                    ellipse_x = state_mean[i, 0] + state_std[i, 0] * np.cos(theta)
                    ellipse_y = state_mean[i, 1] + state_std[i, 1] * np.sin(theta)
                    ax1.fill(ellipse_x, ellipse_y, 'lightblue', alpha=0.1)
            
            ax1.set_xlabel('X Position [m]')
            ax1.set_ylabel('Y Position [m]')
            ax1.set_title('2D Trajectory: Mean ± Std', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Individual robot trajectories (sample)
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(target_mean[:, 0], target_mean[:, 1], 'r-', label='Target', linewidth=3, alpha=0.8)
            for i in range(min(4, num_samples_h)):
                ax2.plot(state_np[:, i, 0], state_np[:, i, 1], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax2.plot(state_mean[:, 0], state_mean[:, 1], 'b-', label='Mean', linewidth=2, alpha=1.0)
            ax2.set_xlabel('X Position [m]')
            ax2.set_ylabel('Y Position [m]')
            ax2.set_title('Individual Robot Trajectories', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            # Position error statistics over time
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(time_np, pos_error_mean, 'g-', linewidth=2, label='Mean Error')
            ax3.fill_between(time_np, 
                           pos_error_mean - pos_error_std, 
                           pos_error_mean + pos_error_std, 
                           alpha=0.3, color='green', label='±1 Std')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Position Error [m]')
            ax3.set_title('Position Tracking Error', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # X position components
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(time_np, target_mean[:, 0], 'r--', label='Target X', linewidth=2, alpha=0.8)
            for i in range(min(10, num_samples_h)):
                ax4.plot(time_np, state_np[:, p+i*10, 0], 
                         color=colors[i % len(colors)], linewidth=1.5, alpha=1, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax4.plot(time_np, state_mean[:, 0], 'r-', label='Mean Actual X', linewidth=2)
            ax4.fill_between(time_np, 
                          state_mean[:, 0] - state_std[:, 0], 
                          state_mean[:, 0] + state_std[:, 0], 
                          alpha=0.3, color='red', label='X ±1 Std')
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('X Position [m]')
            ax4.set_title('X Position', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Y position components
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(time_np, target_mean[:, 1], 'g--', label='Target Y', linewidth=2, alpha=0.8)
            for i in range(min(10, num_samples_h)):
                ax5.plot(time_np, state_np[:, p+i*10, 1], 
                         color=colors[i % len(colors)], linewidth=1.5, alpha=1, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax5.plot(time_np, state_mean[:, 1], 'g-', label='Mean Actual Y', linewidth=2)
            ax5.fill_between(time_np, 
                          state_mean[:, 1] - state_std[:, 1], 
                          state_mean[:, 1] + state_std[:, 1], 
                          alpha=0.3, color='green', label='Y ±1 Std')
            ax5.set_xlabel('Time [s]')
            ax5.set_ylabel('Y Position [m]')
            ax5.set_title('Y Position', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Z position comparison
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(time_np, target_mean[:, 2], 'b--', label='Target Z', linewidth=2, alpha=0.8)
            for i in range(min(10, num_samples_h)):
                ax6.plot(time_np, state_np[:, p+i*10, 2], 
                         color=colors[i % len(colors)], linewidth=1.5, alpha=1, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax6.plot(time_np, state_mean[:, 2], 'b-', label='Mean Actual Z', linewidth=2)
            ax6.fill_between(time_np, 
                           state_mean[:, 2] - state_std[:, 2], 
                           state_mean[:, 2] + state_std[:, 2], 
                           alpha=0.3, color='blue', label='Z ±1 Std')
            ax6.set_xlabel('Time [s]')
            ax6.set_ylabel('Z Position [m]')
            ax6.set_title('Z Position', fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/position_analysis_sce_'+ str(args['scenario'])  + ('_rnn' if args['rnn'] else '_ffn') + (str(args['robot_num']) if args['robot_num'] != -1 else '') + '.png', dpi=300, bbox_inches='tight')
            
            # === FIGURE 2: VELOCITY ANALYSIS ===
            fig_vel = plt.figure(figsize=(18, 8))
            fig_vel.suptitle(f'Velocity Analysis - Scenario {args["scenario"]} {"(RNN)" if args["rnn"] else "(FFN)"}', fontsize=20, fontweight='bold')
            
            # X velocity components
            ax7 = plt.subplot(1, 3, 1)
            for i in range(min(30, num_samples_h)):
                ax7.plot(time_np, state_np[:, p+i*10, 7], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax7.plot(time_np, state_mean[:, 7], 'r-', label='Mean Vel X', linewidth=2)
            ax7.fill_between(time_np, 
                           state_mean[:, 7] - state_std[:, 7], 
                           state_mean[:, 7] + state_std[:, 7], 
                           alpha=0.3, color='red', label='X ±1 Std')
            ax7.set_xlabel('Time [s]')
            ax7.set_ylabel('X Velocity [m/s]')
            ax7.set_title('X Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # Y velocity components
            ax8 = plt.subplot(1, 3, 2)
            for i in range(min(30, num_samples_h)):
                ax8.plot(time_np, state_np[:, p+i*10, 8], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax8.plot(time_np, state_mean[:, 8], 'g-', label='Mean Vel Y', linewidth=2)
            ax8.fill_between(time_np, 
                           state_mean[:, 8] - state_std[:, 8], 
                           state_mean[:, 8] + state_std[:, 8], 
                           alpha=0.3, color='green', label='Y ±1 Std')
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel('Y Velocity [m/s]')
            ax8.set_title('Y Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

            # Z velocity comparison
            ax9 = plt.subplot(1, 3, 3)
            for i in range(min(30, num_samples_h)):
                ax9.plot(time_np, state_np[:, p+i*10, 9], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax9.plot(time_np, state_mean[:, 9], 'b-', label='Mean Vel Z', linewidth=2)
            ax9.fill_between(time_np, 
                           state_mean[:, 9] - state_std[:, 9], 
                           state_mean[:, 9] + state_std[:, 9], 
                           alpha=0.3, color='blue', label='Z ±1 Std')
            ax9.set_xlabel('Time [s]')
            ax9.set_ylabel('Z Velocity [m/s]')
            ax9.set_title('Z Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/velocity_analysis_sce_'+ str(args['scenario'])  + ('_rnn' if args['rnn'] else '_ffn') + (str(args['robot_num']) if args['robot_num'] != -1 else '') + '.png', dpi=300, bbox_inches='tight')

            # === FIGURE 4: ANGLES ANALYSIS ===
            def euler_xyz_from_quat(quat: torch.Tensor):
                qw = quat[:, 3]
                qx = quat[:, 0]
                qy = quat[:, 1]
                qz = quat[:, 2]
                roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                pitch = torch.asin(2.0 * (qw * qy - qz * qx))
                yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                return torch.stack([roll, pitch, yaw], dim=1)
            angles = euler_xyz_from_quat(torch.from_numpy(state_np[:, :, 3:7].reshape(-1, 4))).numpy().reshape(state_np.shape[0], state_np.shape[1], 3)
            fig_angles = plt.figure(figsize=(18, 8))
            fig_angles.suptitle(f'Angles Analysis - Scenario {args["scenario"]} {"(RNN)" if args["rnn"] else "(FFN)"}', fontsize=20, fontweight='bold')
            
            # X angle components
            ax7 = plt.subplot(1, 3, 1)
            for i in range(min(30, num_samples_h)):
                ax7.plot(time_np, angles[:, p+i*10, 0], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax7.plot(time_np, angles[:, :, 0].mean(axis=1), 'r-', label='Mean Angle X', linewidth=2)
            ax7.set_xlabel('Time [s]')
            ax7.set_ylabel('X Angle [rad]')
            ax7.set_title('X Angle: Mean ± Std', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # Y angle components
            ax8 = plt.subplot(1, 3, 2)
            for i in range(min(30, num_samples_h)):
                ax8.plot(time_np, angles[:, p+i*10, 1], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax8.plot(time_np, angles[:, :, 1].mean(axis=1), 'g-', label='Mean Angle Y', linewidth=2)
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel('Y Angle [rad]')
            ax8.set_title('Y Angle: Mean ± Std', fontsize=14, fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

            # Z angle components
            ax9 = plt.subplot(1, 3, 3)
            for i in range(min(30, num_samples_h)):
                ax9.plot(time_np, angles[:, p+i*10, 2], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax9.plot(time_np, angles[:, :, 2].mean(axis=1), 'b-', label='Mean Angle Z', linewidth=2)
            ax9.set_xlabel('Time [s]')
            ax9.set_ylabel('Z Angle [rad]')
            ax9.set_title('Z Angle: Mean ± Std', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/angle_analysis_sce_'+ str(args['scenario'])  + ('_rnn' if args['rnn'] else '_ffn') + (str(args['robot_num']) if args['robot_num'] != -1 else '') + '.png', dpi=300, bbox_inches='tight')

            # === FIGURE 3: ANGULAR VELOCITY ANALYSIS ===
            fig_angvel = plt.figure(figsize=(18, 8))
            fig_angvel.suptitle(f'Angular Velocity Analysis - Scenario {args["scenario"]} {"(RNN)" if args["rnn"] else "(FFN)"}', fontsize=20, fontweight='bold')

            # X angular velocity components
            ax10 = plt.subplot(1, 3, 1)
            for i in range(min(30, num_samples_h)):
                ax10.plot(time_np, state_np[:, p+i*10, 10], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax10.plot(time_np, state_mean[:, 10], 'r-', label='Mean AngVel X', linewidth=2)
            ax10.fill_between(time_np, 
                           state_mean[:, 10] - state_std[:, 10], 
                           state_mean[:, 10] + state_std[:, 10], 
                           alpha=0.3, color='red', label='X ±1 Std')
            ax10.set_xlabel('Time [s]')
            ax10.set_ylabel('X Angular Velocity [rad/s]')
            ax10.set_title('X Angular Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax10.legend()
            ax10.grid(True, alpha=0.3)

            # Y angular velocity components
            ax11 = plt.subplot(1, 3, 2)
            for i in range(min(30, num_samples_h)):
                ax11.plot(time_np, state_np[:, p+i*10, 11], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax11.plot(time_np, state_mean[:, 11], 'g-', label='Mean AngVel Y', linewidth=2)
            ax11.fill_between(time_np, 
                           state_mean[:, 11] - state_std[:, 11], 
                           state_mean[:, 11] + state_std[:, 11], 
                           alpha=0.3, color='green', label='Y ±1 Std')
            ax11.set_xlabel('Time [s]')
            ax11.set_ylabel('Y Angular Velocity [rad/s]')
            ax11.set_title('Y Angular Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax11.legend()
            ax11.grid(True, alpha=0.3)

            # Z angular velocity comparison
            ax12 = plt.subplot(1, 3, 3)
            for i in range(min(30, num_samples_h)):
                ax12.plot(time_np, state_np[:, p+i*10, 12], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax12.plot(time_np, state_mean[:, 12], 'b-', label='Mean AngVel Z', linewidth=2)
            ax12.fill_between(time_np, 
                           state_mean[:, 12] - state_std[:, 12], 
                           state_mean[:, 12] + state_std[:, 12], 
                           alpha=0.3, color='blue', label='Z ±1 Std')
            ax12.set_xlabel('Time [s]')
            ax12.set_ylabel('Z Angular Velocity [rad/s]')
            ax12.set_title('Z Angular Velocity: Mean ± Std', fontsize=14, fontweight='bold')
            ax12.legend()
            ax12.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/angular_velocity_analysis_sce_'+ str(args['scenario'])  + ('_rnn' if args['rnn'] else '_ffn') + (str(args['robot_num']) if args['robot_num'] != -1 else '') + '.png', dpi=300, bbox_inches='tight')
            
            # === FIGURE 5: ACTION ANALYSIS ===
            fig_actions = plt.figure(figsize=(18, 8))
            fig_actions.suptitle(f'Actions Analysis - Scenario {args["scenario"]} {"(RNN)" if args["rnn"] else "(FFN)"}', fontsize=20, fontweight='bold')

            # X angular velocity components
            ax10 = plt.subplot(2, 3, 1)
            for i in range(min(1, num_samples_h)):
                ax10.plot(time_np, action_np[:, p+i*10, 0], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax10.set_ylim(0.0, 1.7)
            ax10.set_xlabel('Time [s]')
            ax10.set_ylabel('Action Value')
            ax10.set_title('Action Analysis 0', fontsize=14, fontweight='bold')
            ax10.legend()
            ax10.grid(True, alpha=0.3)

            # Y angular velocity components
            ax11 = plt.subplot(2, 3, 2)
            for i in range(min(1, num_samples_h)):
                ax11.plot(time_np, action_np[:, p+i*10, 1], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax11.set_ylim(0.0, 1.7)
            ax11.set_xlabel('Time [s]')
            ax11.set_ylabel('Action Value')
            ax11.set_title('Action Analysis 1', fontsize=14, fontweight='bold')
            ax11.legend()
            ax11.grid(True, alpha=0.3)

            # Z angular velocity comparison
            ax12 = plt.subplot(2, 3, 3)
            for i in range(min(1, num_samples_h)):
                ax12.plot(time_np, action_np[:, p+i*10, 2], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax12.set_ylim(0.0, 1.7)
            ax12.set_xlabel('Time [s]')
            ax12.set_ylabel('Action Value')
            ax12.set_title('Action Analysis 2', fontsize=14, fontweight='bold')
            ax12.legend()
            ax12.grid(True, alpha=0.3)

            ax13 = plt.subplot(2, 3, 4)
            for i in range(min(1, num_samples_h)):
                ax13.plot(time_np, action_np[:, p+i*10, 3], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax13.set_ylim(0.0, 1.7)
            ax13.set_xlabel('Time [s]')
            ax13.set_ylabel('Action Value')
            ax13.set_title('Action Analysis 3', fontsize=14, fontweight='bold')
            ax13.legend()
            ax13.grid(True, alpha=0.3)

            # Y angular velocity components
            ax14 = plt.subplot(2, 3, 5)
            for i in range(min(1, num_samples_h)):
                ax14.plot(time_np, action_np[:, p+i*10, 4], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax14.set_ylim(0.0, 1.7)
            ax14.set_xlabel('Time [s]')
            ax14.set_ylabel('Action Value')
            ax14.set_title('Action Analysis 4', fontsize=14, fontweight='bold')
            ax14.legend()
            ax14.grid(True, alpha=0.3)

            # Z angular velocity comparison
            ax15 = plt.subplot(2, 3, 6)
            for i in range(min(1, num_samples_h)):
                ax15.plot(time_np, action_np[:, p+i*10, 5], 
                         color=colors[i % len(colors)], linewidth=1, alpha=0.7, 
                         label=f'Robot {i+1}' if i < 3 else '')
            ax15.set_ylim(0.0, 1.7)
            ax15.set_xlabel('Time [s]')
            ax15.set_ylabel('Action Value')
            ax15.set_title('Action Analysis 5', fontsize=14, fontweight='bold')
            ax15.legend()
            ax15.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/action_analysis_sce_'+ str(args['scenario'])  + ('_rnn' if args['rnn'] else '_ffn') + (str(args['robot_num']) if args['robot_num'] != -1 else '') + '.png', dpi=300, bbox_inches='tight')
            
            # Show all figures
            plt.show(block=False)
            
            # Print comprehensive tracking statistics
            print("\n" + "="*80)
            print(f"HOVER TRAJECTORY TRACKING STATISTICS (n={num_samples_h} robots)")
            print("="*80)
            print(f"Mean position error: {np.mean(pos_error_mean):.4f} ± {np.mean(pos_error_std):.4f} m")
            print(f"Max position error: {np.max(pos_error_mean):.4f} m")
            print(f"Min position error: {np.min(pos_error_mean):.4f} m")
            
            # Calculate tracking performance metrics
            rmse_mean = np.sqrt(np.mean(pos_error_mean**2))
            rmse_std = np.sqrt(np.mean(pos_error_std**2))
            print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} m")
            
            # Position variance statistics
            final_pos_std = [state_std[-1, 0], state_std[-1, 1], state_std[-1, 2]]
            print(f"Final position std (X,Y,Z): [{final_pos_std[0]:.4f}, {final_pos_std[1]:.4f}, {final_pos_std[2]:.4f}] m")
            print(f"Max position std (X,Y,Z): [{np.max(state_std[:,0]):.4f}, {np.max(state_std[:,1]):.4f}, {np.max(state_std[:,2]):.4f}] m")
            print("="*80)


    

if __name__ == "__main__":
    main()
# This script initializes the environment, loads the model, and runs a simulation loop.
# It prints the reward and done status at each step.
# Make sure to adjust the paths and configurations according to your setup.
# You can run this script to test the RL-Games player with your custom environment.