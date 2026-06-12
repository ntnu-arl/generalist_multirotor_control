import numpy as np
import os
import yaml

import gym
from gym import spaces
from rl_games.common import env_configurations, vecenv

import torch
import distutils
from distutils import util

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"



#### Custom setup ######
import pickle as pkl
import generalist_multirotor_control as mds
package_prefix = os.path.dirname(mds.__file__) + "/"
filename1 = package_prefix + "airframe_generation/valid_airframe_config_6_all.pkl"



with open(filename1, "rb") as f:
    loaded_configs = pkl.load(f)
#######


#######

from generalist_multirotor_control.rl_training.rl_games.networks.gen_policy_network import RnnTestNetBuilder, DecTestNetBuilder
from rl_games.algos_torch import model_builder
model_builder.register_network('rnntestnet', RnnTestNetBuilder)
model_builder.register_network('dectestnet', DecTestNetBuilder)
#######


from generalist_multirotor_control.simulator.multirotor_dynamics_env import MultirotorDynamicsEnv as DynamicsGymInterface
from generalist_multirotor_control.simulator.custom_obs_env import CustomObsEnv

class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations #["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        return (
            observations, #["observations"],
            rewards,
            dones,
            infos,
        )


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
            # np.ones(13) * -np.Inf,
            # np.ones(13) * np.Inf,
            # np.ones(37) * -np.Inf,
            # np.ones(37) * np.Inf,
        )
        print(info["action_space"], info["observation_space"])
        return info




vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
         {
            "name": "--robot_num",
            "type": int,
            "default": -1,
            "required": False,
            "help": "Configuration id to train for.",
        },
        {
            "name": "--scenario",
            "type": int,
            "default": -1,
            "required": False,
            "help": "Configuration id to train for.",
        },
         {
            "name": "--langevin",
            "help": "Whether to use langevin dynamics.",
            "action": "store_true",
        },
        {
            "name": "--rnn",
            "help": "Whether to use RNN.",
            "action": "store_true",
        },

         {
            "name": "--importance_sampling",
            "help": "Whether to use importance sampling.",
            "action": "store_true",
        },
        {
            "name": "--randomize_descriptor",
            "help": "Whether to use descriptor randomization.",
            "action": "store_true",
        },
        {
            "name": "--tmp",
            "help": "Whether to store temporary files.",
            "default": False,
            "action": "store_true",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "2048",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "generalizable_policy_training",
            "help": "the wandb's project name",
        },
        {
            "name": "--task",
            "type": str,
            "default": "position_setpoint_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
    ]

    import argparse
    parser = argparse.ArgumentParser(description="Multirotor Dynamics Simulator")
    for argument in custom_parameters:
        parser.add_argument(argument["name"], **{k: v for k, v in argument.items() if k != "name"})
    # parse arguments
    parsed_args, unknown_args = parser.parse_known_args()
    print("[runner] Unknown args: ", unknown_args)
    return parsed_args


def update_config(config, args):

    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    config_name = args["file"]

    print("Loading config: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)

        config = update_config(config, args)

        from rl_games.torch_runner import Runner

        runner = Runner()
        try:
            print("Loading config...", config)
            print("Simulating with robot number: ", config.keys())
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    rank = 0 #int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb
        wandb.login(key="77c03fddc407c79d5db797b85850a31b3a822af5")
        print("Tracking this run with wandb")
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
            name=args["experiment_name"]
        )
    #for i in range(10):
    #    print('condiguration: ', i)
    loaded_configs["robot_num"] = args.get('robot_num', -1) # set robot number here
    loaded_configs["scenario"] = args.get('scenario', -1) # set scenario here
    loaded_configs["langevin"] = args.get('langevin', False) # whether to use langevin dynamics
    loaded_configs["importance_sampling"] = args.get('importance_sampling', False)
    loaded_configs["randomize_descriptor"] = None #args.get('randomize_descriptor', None)
    args["checkpoint"] = config["params"].get("checkpoint", None)
    env_name = f"position_setpoint_task_custom"
    env_configurations.register(
        "position_setpoint_task_custom",
        {
            "env_creator": lambda **kwargs: CustomObsEnv(config=loaded_configs, device="cuda:0"),
            "vecenv_type": "AERIAL-RLGPU",
        },
    )
    print("Initial config: ", config)
    config["params"]["config"]["env_name"] = env_name
    if args["rnn"]:
        print('Using RNN policy network')
        config["params"]["network"]['name'] = "rnntestnet"
    else:
        print('Using Feedforward policy network')
        config["params"]["network"]['name'] = "dectestnet"
    print("Final config: ", config)
    runner.load(config)
    runner.run(args)
    


    if args["track"] and rank == 0:
        wandb.finish()
