from generalist_multirotor_control.utils.math_utils import euler_angles_to_matrix, quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from generalist_multirotor_control.utils.math_utils import (
    quat_rotate_inverse, quat_rotate_multidim
)
from generalist_multirotor_control.simulator.multirotor_dynamics_env import MultirotorDynamicsEnv
import numpy as np
import os

from collections import deque

log_states = deque(maxlen=10)
log_resets = deque(maxlen=10)

import torch
import warp as wp

DEBUG_MODE = False

global NUM_ENVS
NUM_ENVS=2048 #2048 #2048 #2048
#ROBOT_NUM = -1


# ----- DO NOT TOUCH BELOW THIS LINE -----

ENABLE_VALID_AMAT = False
USE_NORMALIZED_ALLOCATION_MATRIX = False
ENABLE_PINVERSE = True
USE_NORMALIZED_FORCE = False
MASS_AND_INERTIA_OBSERVATIONS = False
USE_MOTOR_ANGVEL = False
FORCE_AND_TIME_CONSTANTS = False
MOTOR_POSES = False

# ----- DO NOT TOUCH ABOVE THIS LINE -----

print('CUDA_VISIBLE_DEVICES:', os.environ.get("CUDA_VISIBLE_DEVICES"))

def apply_scenario(scenario):
    global ENABLE_VALID_AMAT, USE_NORMALIZED_ALLOCATION_MATRIX, MOTOR_POSES
    global USE_NORMALIZED_FORCE, MASS_AND_INERTIA_OBSERVATIONS
    global ENABLE_PINVERSE, FORCE_AND_TIME_CONSTANTS, USE_MOTOR_ANGVEL

    if scenario == 1:
        print("Scenario 1: No robot info (defaults).")
        USE_NORMALIZED_FORCE = True

    elif scenario == 2:
        print("Scenario 2: Only allocation matrix.")
        ENABLE_VALID_AMAT = True
        USE_NORMALIZED_FORCE = True


    elif scenario == 3:
        print("Scenario 3: Allocation matrix + mass/inertia.")
        ENABLE_VALID_AMAT = True
        MASS_AND_INERTIA_OBSERVATIONS = True
        USE_NORMALIZED_FORCE = True

    elif scenario == 4:
        print("Scenario 4: Normalized allocation matrix + pinverse.")
        ENABLE_VALID_AMAT = True
        USE_NORMALIZED_ALLOCATION_MATRIX = True
        USE_NORMALIZED_FORCE = True
        ENABLE_PINVERSE = False

    elif scenario == 5:
        print("Scenario 5: Make the problem MDP. Use normalized allocation matrix, motor angvels and force/time constants.")
        ENABLE_VALID_AMAT = True
        USE_NORMALIZED_ALLOCATION_MATRIX = True
        USE_NORMALIZED_FORCE = True
        ENABLE_PINVERSE = False
        FORCE_AND_TIME_CONSTANTS = True
        USE_MOTOR_ANGVEL = True

    elif scenario == 6:
        print("Scenario 6: Use allocation matrix, mass/inertia, motor angvels and force/time constants.")
        ENABLE_VALID_AMAT = True
        MASS_AND_INERTIA_OBSERVATIONS = False
        USE_NORMALIZED_FORCE = True
        FORCE_AND_TIME_CONSTANTS = True
        USE_MOTOR_ANGVEL = True

    elif scenario == 7:
        print("Scenario 7: Allocation matrix, mass and inertia, force/time constants.")
        ENABLE_VALID_AMAT = True
        MASS_AND_INERTIA_OBSERVATIONS = True
        FORCE_AND_TIME_CONSTANTS = True
        USE_NORMALIZED_FORCE = True
        ENABLE_PINVERSE = False
    elif scenario == 8:
        print("Scenario 8: Normalized allocation matrix, force/time constants.")
        ENABLE_VALID_AMAT = True
        FORCE_AND_TIME_CONSTANTS = True
        USE_NORMALIZED_ALLOCATION_MATRIX = True
        USE_NORMALIZED_FORCE = True
        ENABLE_PINVERSE = False

    elif scenario == 9:
        print("Scenario 9: Normalized allocation matrix, force/time constants, pinverse.")
        ENABLE_VALID_AMAT = True
        FORCE_AND_TIME_CONSTANTS = True
        USE_NORMALIZED_ALLOCATION_MATRIX = True
        USE_NORMALIZED_FORCE = True
        ENABLE_PINVERSE = True

    elif scenario == 10:
        print("Scenario 10: Motor poses")
        MOTOR_POSES = True
        FORCE_AND_TIME_CONSTANTS = True
        MASS_AND_INERTIA_OBSERVATIONS = True
        USE_NORMALIZED_FORCE = True

    elif scenario == 11:
        USE_MOTOR_ANGVEL = True
        USE_NORMALIZED_FORCE = True
        print("Scenario 11: No robot info (defaults), but motor rpms.")


    else:
        raise ValueError(f"Unknown scenario: {scenario}")



class CustomObsEnv(MultirotorDynamicsEnv):
    def __init__(self, config, seed=None, device="cuda:0"):
        print("CustomObsEnv init called, config: ", config.keys())
        if seed is not None:
            if seed < 0:
                import time
                seed = time.time_ns() % 2**32
            else:
                seed = int(seed)
            print("Seed: ", seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        self.config = config
        self.device = device
        self.state_dict = None
        self.force_tensor_dict = None
        self.root_state_tensor = None
        self.motor_rpm_tensor = None
        self.motor_setpoint_tensor = None
        self.motor_direction_tensor = None
        self.motor_force_constant_tensor = None
        self.motor_time_constant_tensor = None
        self.mass_tensor = None
        self.inertia_tensor = None
        self.inertia_inverse_tensor = None
        self.gravity_tensor = None
        self.allocation_matrix_tensor = None
        self.dt = config.get("dt", 0.01)
        self.euler_graph = None
        self.rk4_graph = None
        self.counter_tensor = None
        self.info = {}
        self.max_episode_steps = config.get("max_episode_steps", 600)
        self.crash_dict = None
        self.step_counter = 0
        self.train_epochs = 0
        self.reset_counter = 0
        self.ROBOT_NUM = config.get("robot_num", -1)
        self.scenario = config.get("scenario", 1)
        self.randomize_descriptor = config.get("randomize_descriptor", None)
        if self.randomize_descriptor is not None: 
            self.randomize_descriptor = {
                'mass_rel_std': 0.1,
                'inertia_rel_std': 0.1,
                'motor_pos_randomize_range': 0.01,
                'torque_randomize_range': 0.1,
                'motor_orientation_randomize_range': torch.pi / 45, # about 4 degrees
                'force_randomize_range': 0.1,
                'time_randomize_range': 0.1
            }
            print("Using randomization with descriptor: ", self.randomize_descriptor)
        self.follow_trajectory = False
        self.langevin = config.get("langevin", False)
        self.lissajous = config.get("lissajous", False)
        self.importance_sampling = config.get('importance_sampling', False)
        self.noisy = config.get("noisy", False)
        self.obs_lpf_alpha_angvel = 0.0
        self.state_noise = {'pos': 0.001, 'ori': torch.pi/1024, 'linvel': 0.002, 'angvel': 0.001} 
        self.one_point = config.get("one_point", False)
        self.train_individual = config.get("train_individual", False)
        self.repeat_configs = config.get("repeat_configs", 1)
        global NUM_ENVS
        if self.repeat_configs > 1 and not self.train_individual:
            NUM_ENVS = 2560 
        if self.train_individual:
                print("Training batched specialists policies.")
        print("Number of envs: ", NUM_ENVS)
        apply_scenario(self.scenario)
        if self.langevin:
            print('env is langevin')
            self.follow_trajectory = True
            self.p = 0.5   # probability of using langevin dynamics
        elif self.lissajous:
            print('env is lissajous')
            self.follow_trajectory = True
            self.p = 1.0 #0.5   # probability of using lissajous dynamics
            self.same = config.get("same", True)

        self.init_tensors_from_dict(self.config)
        self.enable_obs_lpf = False
        self.delay_noise, self.delay_prob = False, 0.5

        if self.enable_obs_lpf or self.delay_noise:
            # LPF coefficients
            # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
            self.obs_lpf_alpha_pos = 1.0
            self.obs_lpf_alpha_ori = 1.0
            self.obs_lpf_alpha_linvel = 1.0
            self.obs_lpf_alpha_angvel = 0.4

            # Filter states
            self.obs_pos_lpf = torch.zeros((self.num_envs, 3), device=self.device)
            self.obs_euler_lpf = torch.zeros((self.num_envs, 3), device=self.device)
            self.obs_linvel_lpf = torch.zeros((self.num_envs, 3), device=self.device)
            self.obs_angvel_lpf = torch.zeros((self.num_envs, 3), device=self.device)

            # Helps initialize filter without a startup transient
            self.obs_lpf_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.capture_graphs()
        # print reset_logger
        self.env_reset_counter = torch.zeros(self.num_envs, device=self.device)
        self.begin_counting_resets_at_ts = 3400
        self.first = 0 # first time to reset
        self.N, self.reward_mean, self.reward_std = 0, torch.zeros(self.num_envs, device=self.device), torch.ones(self.num_envs, device=self.device)
        self.rw2cfg_permutation = torch.arange(0, self.num_envs, device=self.device)
        self.samples_per_cfg = np.zeros(self.num_envs, dtype=np.int32)
        self.angvel_error_body_frame_old = torch.zeros((self.num_envs, 3), device=self.device)

        self.reset()

    def __del__(self):
        self.close_sim()

    def close_sim(self):
        print("Destructor called")
        print("Total steps: ", self.step_counter)
        print("Worst 10 robots: ", torch.topk(self.env_reset_counter, 10, largest=True))
        print("Best 10 robots: ", torch.topk(self.env_reset_counter, 10, largest=False))

        # print mean and std
        print("Mean: ", torch.mean(self.env_reset_counter))
        print("Std: ", torch.std(self.env_reset_counter))
        # write the IDS to file that are lesser than mean
        print("less than mean counter: ", torch.sum(self.env_reset_counter < torch.mean(self.env_reset_counter)))
        # print("Less than mean: ", torch.nonzero(self.env_reset_counter < torch.mean(self.env_reset_counter)))

       
    def step(self, actions, reward_params=None):
        # unnormalize motor forces => forces = robot_mass * actions
        #print('step actions before clamp:', actions[3])
        #actions = torch.rand_like(actions) 
        if USE_NORMALIZED_FORCE:
            actions = (torch.clamp(actions, -1.0, 1.0) + 1.0) / 2.0 # normalize actions to [0, 1]
            force_metric = (self.mass_tensor * 9.81) * actions /2 # 0.75*m*g for quadrotors 
        else:
            force_metric = 12.25 * actions / 2
        ret = super().step(actions=force_metric, metric_force=True)
        
        return ret



    def reset_idx(self, env_ids):
        if self.step_counter > self.begin_counting_resets_at_ts:
            self.env_reset_counter[env_ids] += 1
        if self.follow_trajectory:
            if self.langevin:
                self.traj_pos[env_ids] = 0.0
                self.traj_vel[env_ids] = 0.0
                self.traj_pos_raw[env_ids] = 0.0
                self.traj_vel_raw[env_ids] = 0.0
            elif self.lissajous:
                ids = env_ids[(self.traj[env_ids.cpu()] == 1)]
                local_ids = self.global_to_traj[ids].cpu()
                self.traj_pos[ids] = self.lissajous_traj['pos'][local_ids, self.lis_time]
                self.traj_vel[ids] = self.lissajous_traj['vel'][local_ids, self.lis_time]

        if self.importance_sampling and self.first>500:
            if env_ids.shape[0] > 0:
                new_robot_config, sampled_indices = self.get_new_config(env_ids.shape[0])
                mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                mask[env_ids] = True
                self.rw2cfg_permutation[env_ids] = torch.tensor(sampled_indices).to(self.device)
                self.init_tensors_from_dict_v2(loaded_configs=new_robot_config, mask = mask)

        self.first +=1
        return super().reset_idx(env_ids)
    
    def get_new_config(self, n):
        new_robot_config = {}
        new_robot_config['num_robots'] = n
        new_robot_config['num_motors'] = self.config['num_motors']
    
        probs = torch.softmax(1*self.difficulty, dim=0) #4 for quads, 1 for hex
        sampled_indices = torch.multinomial(probs, n, replacement=False).cpu().numpy()
        self.samples_per_cfg[sampled_indices] +=1
        for k, v in self.config.items():
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                new_robot_config[k] = v[sampled_indices]
        return new_robot_config, sampled_indices

    def get_normalized_allocation_matrix_tensor(self, allocation_matrix_tensor, mass_tensor, inertia_tensor, inertia_inverse_tensor):
        """
        Computes the normalized allocation matrix tensor based on the robot mass, inertia, and inertia inverse tensors.
        The normalization is done by dividing the allocation matrix by the square root of the product of mass and inertia.
        """

        B1 = allocation_matrix_tensor[:, 0:3, :]  # 3x4
        B2 = allocation_matrix_tensor[:, 3:6, :]  # 3x4

        B1_norm = B1

        B2_norm = mass_tensor.unsqueeze(2) * inertia_inverse_tensor @ B2  # 3x4

        allocation_matrix_normalized = torch.zeros_like(allocation_matrix_tensor)
        allocation_matrix_normalized[:, 0:3, :] = B1_norm
        allocation_matrix_normalized[:, 3:6, :] = B2_norm
        return allocation_matrix_normalized
    
    def get_randomized_allocation_matrix_tensor(self, motor_poses, motor_directions, motor_torque_constants, motor_pos_randomize_range = 0.001, torque_randomize_range = 0.002, motor_orientation_randomize_range = torch.pi / 1024):
        device = motor_poses.device
        # Use the passed parameters instead of hard-coded values
        # torque_randomize_range, motor_pos_randomize_range = 0.002, 0.05  # REMOVED: hard-coded override

        R, M = motor_poses.shape[0], motor_poses.shape[1]
        allocation_matrix = torch.zeros((R, 6, M), dtype=torch.float32, device=device)

        # torque constant noise
        motor_torque_constants = motor_torque_constants*(1.0 + (torch.rand(R, M, device=device) * 2.0 - 1.0) * torque_randomize_range)

        motor_force_frame = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).view(1, 1, 3).expand(R, M, 3)
        motor_torque = motor_force_frame * motor_torque_constants.unsqueeze(-1)

        # motor position noise
        motor_pos = motor_poses[..., 0:3] + (torch.rand_like(motor_poses[..., 0:3]) * 2.0 - 1.0) * motor_pos_randomize_range

        # -------- orientation noise (Euler) --------
        motor_quat_xyzw = motor_poses[..., 3:7]                               # (R,M,4)
        motor_quat_wxyz = motor_quat_xyzw[..., [3, 0, 1, 2]]                  # -> (w,x,y,z)

        # convert to euler; keep dims
        motor_mat = quaternion_to_matrix(motor_quat_wxyz.reshape(-1, 4)).reshape(R, M, 3, 3)

        # matrix_to_euler_angles returns (...,3) in the specified convention ordering
        euler_zyx = matrix_to_euler_angles(motor_mat.reshape(-1,3,3), "ZYX").reshape(R, M, 3)  # (R,M,3)

        # noise in radians per euler component
        euler_noise = torch.normal(mean=torch.zeros((R, M, 3), device=device), std=(motor_orientation_randomize_range))
        euler_noisy = euler_zyx + euler_noise

        noisy_mat = euler_angles_to_matrix(euler_noisy.reshape(-1,3), "ZYX").reshape(R, M, 3, 3)
        noisy_quat_wxyz = matrix_to_quaternion(noisy_mat.reshape(-1,3,3)).reshape(R, M, 4)
        motor_quat = noisy_quat_wxyz[..., [1, 2, 3, 0]]                       # back to xyzw
        # ------------------------------------------

        motor_force_in_robot_frame = quat_rotate_multidim(motor_quat, motor_force_frame)
        motor_torque_in_robot_frame = quat_rotate_multidim(motor_quat, motor_torque *(- motor_directions.to(device)))

        torque = torch.cross(motor_pos, motor_force_in_robot_frame, dim=-1) + motor_torque_in_robot_frame

        allocation_matrix[:, 0:3, :] = motor_force_in_robot_frame.transpose(1, 2)
        allocation_matrix[:, 3:6, :] = torque.transpose(1, 2)
        return allocation_matrix


    def init_tensors_from_dict(self, loaded_configs=None, global_tensor_dict=None):
        print('RANDOMIZE DESCRIPTOR: ', self.randomize_descriptor)
        if global_tensor_dict is None:
            global_tensor_dict = {}
        if loaded_configs is None:
            raise ValueError("Config is None. Needs robot information.")

        loaded_configs["num_robots"] = int(loaded_configs["num_robots"])
        num_envs = NUM_ENVS #loaded_configs["num_robots"]
        num_motors = loaded_configs["num_motors"]
        print("Number of envs: ", num_envs)
        print("Number of motors: ", num_motors)
        self.num_envs = num_envs
        self.num_motors = num_motors

      
        if self.ROBOT_NUM >= 0:
            print("LOADING BASED ON ROBOT NUMBER. ROBOT_NUM: ", self.ROBOT_NUM)
            print("Simulating with robot number: ", self.ROBOT_NUM)
            valid_robot_mass = torch.tensor(loaded_configs["robot_mass"]).unsqueeze(-1)[self.ROBOT_NUM].expand(NUM_ENVS, 1)
            valid_robot_inertias = torch.tensor(loaded_configs["robot_inertia"])[self.ROBOT_NUM].expand(NUM_ENVS, 3, 3)
            valid_robot_inertias_inverse = torch.tensor(loaded_configs["robot_inertia_inverse"])[self.ROBOT_NUM].expand(NUM_ENVS, 3, 3)
            valid_allocation_matrices = torch.tensor(loaded_configs["allocation_matrix"])[self.ROBOT_NUM].expand(NUM_ENVS, 6, self.num_motors)
            valid_allocation_matrices_pinverse = torch.tensor(loaded_configs["allocation_matrix_inverse"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 6)
            valid_motor_max_thrusts = torch.tensor(loaded_configs["motor_max_thrusts"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors)
            valid_motor_min_thrusts = torch.tensor(loaded_configs["motor_min_thrusts"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors)
            valid_motor_time_constants = torch.tensor(loaded_configs["motor_time_constants"]).unsqueeze(-1)[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 1)
            valid_motor_force_constants = torch.tensor(loaded_configs["motor_force_constants"]).unsqueeze(-1)[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 1)
            valid_motor_directions = torch.tensor(loaded_configs["motor_directions"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 1)
            valid_motor_poses = torch.tensor(loaded_configs["motor_poses"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 7)
            valid_motor_poses_com = torch.tensor(loaded_configs["motor_poses_com"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 7)
            valid_motor_torque_constants = torch.tensor(loaded_configs["motor_torque_constants"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors)
            valid_motor_masses = torch.tensor(loaded_configs["motor_mass"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors)
            valid_motor_inertias = torch.tensor(loaded_configs["motor_inertia"])[self.ROBOT_NUM].expand(NUM_ENVS, self.num_motors, 3, 3)
        else:
            print("LOADING ALL ROBOTS")
            valid_robot_mass = torch.tensor(loaded_configs["robot_mass"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            print("valid_robot_mass shape: ", valid_robot_mass.shape, torch.tensor(loaded_configs["robot_mass"]).shape, torch.tensor(loaded_configs["robot_mass"]).repeat_interleave(self.repeat_configs, dim=0).shape)
            valid_robot_inertias = torch.tensor(loaded_configs["robot_inertia"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_robot_inertias_inverse = torch.tensor(loaded_configs["robot_inertia_inverse"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_allocation_matrices = torch.tensor(loaded_configs["allocation_matrix"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_allocation_matrices_pinverse = torch.tensor(loaded_configs["allocation_matrix_inverse"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_max_thrusts = torch.tensor(loaded_configs["motor_max_thrusts"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_min_thrusts = torch.tensor(loaded_configs["motor_min_thrusts"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]

            valid_motor_time_constants = torch.tensor(loaded_configs["motor_time_constants"]).unsqueeze(-1).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS] #
            valid_motor_force_constants = torch.tensor(loaded_configs["motor_force_constants"]).unsqueeze(-1).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_torque_constants = torch.tensor(loaded_configs["motor_torque_constants"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        
            valid_motor_directions = torch.tensor(loaded_configs["motor_directions"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_poses = torch.tensor(loaded_configs["motor_poses"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_poses_com = torch.tensor(loaded_configs["motor_poses_com"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_masses = torch.tensor(loaded_configs["motor_mass"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
            valid_motor_inertias = torch.tensor(loaded_configs["motor_inertia"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]

        self.observation_space_dim = 13 + (self.num_motors * 6 if ENABLE_VALID_AMAT else 0) + (7*self.num_motors if MOTOR_POSES else 0) + (10 if MASS_AND_INERTIA_OBSERVATIONS else 0) + (2*self.num_motors if FORCE_AND_TIME_CONSTANTS else 0) + (self.num_motors if USE_MOTOR_ANGVEL else 0)   # 13 + 24 + 1 + 9 + 1 (conf id)
        self.action_space_dim = self.num_motors
      

        self.root_state_tensor = torch.zeros(self.num_envs, 13, device="cuda:0", requires_grad=False)

        self.actions = torch.zeros(self.num_envs, self.action_space_dim, device="cuda:0", requires_grad=False)
        self.prev_actions = torch.zeros(self.num_envs, self.action_space_dim, device="cuda:0", requires_grad=False)
        self.motor_rpm_tensor = torch.zeros(self.num_envs, self.num_motors, 1, device="cuda:0", requires_grad=False)
        self.motor_setpoint_tensor = torch.zeros(
            self.num_envs, self.num_motors, 1, device="cuda:0", requires_grad=False
        )
        self.motor_direction_tensor = torch.zeros(
            self.num_envs, self.num_motors, 1, device="cuda:0", requires_grad=False
        )
        self.motor_force_constant_tensor = torch.zeros(
            self.num_envs, self.num_motors, 1, device="cuda:0", requires_grad=False
        )
        self.motor_time_constant_tensor = torch.zeros(
            self.num_envs, self.num_motors, 1, device="cuda:0", requires_grad=False
        )
        self.mass_tensor = torch.ones(self.num_envs, 1, device="cuda:0", requires_grad=False)
        self.inertia_tensor = (
            torch.eye(3, device="cuda:0").unsqueeze(0).expand(num_envs, -1, -1).clone()
        )
        self.inertia_inverse_tensor = (
            torch.eye(3, device="cuda:0").unsqueeze(0).expand(num_envs, -1, -1).clone()
        )
        self.gravity_tensor = torch.zeros(num_envs, 3, device="cuda:0", requires_grad=False)
        self.gravity_tensor[:, 2] = -9.81
        self.allocation_matrix_tensor = (
            torch.ones(6, self.num_motors, device="cuda:0", requires_grad=False).expand(num_envs, -1, -1).clone()
        )


         # --- noisy mass/inertia for simulation only ---
        if self.randomize_descriptor is not None:
            print('using randomized descriptor',self.randomize_descriptor)
            #uniform 0.1 mass, 
            mass_rel_std = self.randomize_descriptor['mass_rel_std']   # 2%
            inertia_rel_std = self.randomize_descriptor['inertia_rel_std'] # 5% (tune)

            m_obs = self.rel_noise(valid_robot_mass, mass_rel_std, clamp_min=1e-4)         # (B,1) or (B,)
            J_obs = self.rel_noise(valid_robot_inertias, inertia_rel_std)                   # (B,3,3)
            true_robot_mass = m_obs
            true_robot_inertias = J_obs

            Jinv_obs = torch.linalg.inv(J_obs)  # (B,3,3)
            true_robot_inertias_inverse = Jinv_obs
            true_motor_torque_constants = self.rel_noise(valid_motor_torque_constants, self.randomize_descriptor['torque_randomize_range'])  # (B,M,1)
            true_motor_force_constants = self.rel_noise(valid_motor_force_constants, self.randomize_descriptor['force_randomize_range'])  # (B,M,1)
            true_motor_time_constants = self.rel_noise(valid_motor_time_constants, self.randomize_descriptor['time_randomize_range'], clamp_min=1e-4)  # (B,M,1)

            # --- compute noisy normalized allocation matrix using CLEAN allocation_matrix_tensor ---
            #A_clean = self.allocation_matrix_tensor[mask]  # (B,6,M) or whatever your shape is
            A = self.get_randomized_allocation_matrix_tensor(valid_motor_poses, valid_motor_directions, true_motor_torque_constants,
                                                             self.randomize_descriptor['motor_pos_randomize_range'], self.randomize_descriptor['torque_randomize_range'], 
                                                             self.randomize_descriptor['motor_orientation_randomize_range']).to(self.device)  # (B,6,M)            
            true_allocation_matrices = A

        else:
            true_robot_mass = valid_robot_mass
            true_robot_inertias = valid_robot_inertias
            true_robot_inertias_inverse = valid_robot_inertias_inverse
            true_allocation_matrices = valid_allocation_matrices
            true_motor_force_constants = valid_motor_force_constants
            true_motor_time_constants = valid_motor_time_constants
            true_motor_torque_constants = valid_motor_torque_constants

        self.mass_tensor[:] = true_robot_mass[:]

        self.inertia_tensor[:] = true_robot_inertias[:]
        self.inertia_inverse_tensor[:] = true_robot_inertias_inverse[:]

        self.allocation_matrix_tensor[:] = true_allocation_matrices[:]
        self.motor_direction_tensor[:] = valid_motor_directions[:]

        self.allocation_matrix_tensor_pinv = torch.linalg.pinv(self.allocation_matrix_tensor)
        self.normalized_allocation_matrix = self.get_normalized_allocation_matrix_tensor(
            self.allocation_matrix_tensor, self.mass_tensor, self.inertia_tensor, self.inertia_inverse_tensor
        )
        valid_allocation_matrices_normalized = self.get_normalized_allocation_matrix_tensor(
            valid_allocation_matrices, valid_robot_mass, valid_robot_inertias, valid_robot_inertias_inverse
        )
        valid_allocation_matrices_normalized_pinv = torch.linalg.pinv(valid_allocation_matrices_normalized)

        self.normalized_allocation_matrix_pinverse = torch.linalg.pinv(self.normalized_allocation_matrix)


        self.motor_force_constant_tensor[:] = true_motor_force_constants
        self.motor_time_constant_tensor[:] = true_motor_time_constants
        self.motor_max_thrusts = valid_motor_max_thrusts.to(device=self.device)
        self.motor_min_thrusts = valid_motor_min_thrusts.to(device=self.device)
        self.motor_poses = valid_motor_poses
        self.motor_torque_constants = true_motor_torque_constants
        self.motor_poses_com = valid_motor_poses_com

        self.gravity_tensor[:] = self.gravity_tensor[:]
        self.counter_tensor = torch.zeros(
            num_envs, device="cuda:0", requires_grad=False, dtype=torch.int32
        )
        global_tensor_dict["root_state_tensor"] = self.root_state_tensor
        global_tensor_dict["motor_rpm_tensor"] = self.motor_rpm_tensor
        global_tensor_dict["motor_setpoint_tensor"] = self.motor_setpoint_tensor
        global_tensor_dict["motor_direction_tensor"] = self.motor_direction_tensor
        global_tensor_dict["motor_force_constant_tensor"] = self.motor_force_constant_tensor
        global_tensor_dict["motor_time_constant_tensor"] = self.motor_time_constant_tensor
        global_tensor_dict["mass_tensor"] = self.mass_tensor
        global_tensor_dict["inertia_tensor"] = self.inertia_tensor
        global_tensor_dict["inertia_inverse_tensor"] = self.inertia_inverse_tensor
        global_tensor_dict["gravity_tensor"] = self.gravity_tensor
        global_tensor_dict["allocation_matrix_tensor"] = self.allocation_matrix_tensor
        global_tensor_dict["dt"] = self.dt

        self.pos = self.root_state_tensor[:, 0:3]
        self.quat = self.root_state_tensor[:, 3:7]
        
    
        self.quat[:, 0:3] = 0.0
        self.quat[:, 3] = 1.0
            
        self.vel = self.root_state_tensor[:, 7:10]
        self.angvel = self.root_state_tensor[:, 10:13]
        self.prev_angvel = torch.zeros_like(self.angvel)
        
        if self.follow_trajectory:
            self.traj = torch.bernoulli(torch.full((self.root_state_tensor.size(0),), self.p))
            print('env follows trajectory: ', self.traj.mean().item())
            # All envs with Lissajous
            self.traj_env_ids = torch.nonzero(self.traj == 1, as_tuple=True)[0]  # shape [N_traj]
            # Build a mapping global → local (size = num_envs, fill with -1)
            self.global_to_traj = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
            self.global_to_traj[self.traj_env_ids] = torch.arange(len(self.traj_env_ids), device=self.device)
            if self.langevin:
                self.traj_pos = torch.zeros_like(self.pos, device=self.device)
                self.traj_vel = torch.zeros_like(self.vel, device=self.device)
                self.traj_pos_raw = torch.zeros_like(self.pos, device=self.device)
                self.traj_vel_raw = torch.zeros_like(self.vel, device=self.device)
            elif self.lissajous:
                self.lis_time = 0
                self.traj_pos = torch.zeros_like(self.pos, device=self.device)
                self.traj_vel = torch.zeros_like(self.vel, device=self.device)
                self.lissajous_figure_eight_torch(batch_size=self.traj.sum().int().item(), same=self.same, dt = self.dt)  # initialize traj_pos and traj_vel for envs that follow trajectory
        self.crashes = torch.zeros(self.root_state_tensor.size(0), device=self.device)
        self.rewards = torch.zeros(self.root_state_tensor.size(0), device=self.device)
        self.truncations = torch.zeros(self.root_state_tensor.size(0), device=self.device)

        self.pos_wp = wp.from_torch(self.pos, dtype=wp.vec3f)
        self.quat_wp = wp.from_torch(self.quat, dtype=wp.vec4f)
        self.vel_wp = wp.from_torch(self.vel, dtype=wp.vec3f)
        self.angvel_wp = wp.from_torch(self.angvel, dtype=wp.vec3f)
        self.motor_rpm_wp = wp.from_torch(
            self.motor_rpm_tensor, dtype=wp.mat(shape=(self.num_motors, 1), dtype=wp.float32)
        )
        self.motor_force_constant_wp = wp.from_torch(
            self.motor_force_constant_tensor, dtype=wp.mat(shape=(self.num_motors, 1), dtype=wp.float32)
        )
        self.motor_direction_wp = wp.from_torch(
            self.motor_direction_tensor, dtype=wp.mat(shape=(self.num_motors, 1), dtype=wp.float32)
        )
        self.motor_time_constant_wp = wp.from_torch(
            self.motor_time_constant_tensor, dtype=wp.mat(shape=(self.num_motors, 1), dtype=wp.float32)
        )
        #print('motor_time_constant_tensor:', self.motor_time_constant_tensor[0], self.motor_time_constant_tensor[-2])
        self.mass_wp = wp.from_torch(self.mass_tensor[:, 0], dtype=wp.float32)
        self.inertia_wp = wp.from_torch(self.inertia_tensor, dtype=wp.mat33f)
        self.inertia_inverse_wp = wp.from_torch(self.inertia_inverse_tensor, dtype=wp.mat33f)
        self.gravity_wp = wp.from_torch(self.gravity_tensor, dtype=wp.vec3f)
        self.motor_setpoint_wp = wp.from_torch(
            self.motor_setpoint_tensor, dtype=wp.mat(shape=(self.num_motors, 1), dtype=wp.float32)
        )
        self.amat_wp = wp.from_torch(
            self.allocation_matrix_tensor, dtype=wp.mat(shape=(6, self.num_motors), dtype=wp.float32)
        )
       

        self.observations_tensor = torch.zeros(self.num_envs, self.observation_space_dim, device="cuda:0", requires_grad=False)
        if self.train_individual:
            self.emb_id = (torch.arange(NUM_ENVS, device="cuda:0").float()/NUM_ENVS)  # shape (num_envs,)
            print("emb_id: ", self.emb_id)
        self.index_ = 13
        if USE_MOTOR_ANGVEL:
            self.observations_tensor[:, self.index_:self.index_+self.num_motors] = torch.zeros(self.num_envs, self.num_motors, device="cuda:0")  # motor angvels
            self.index_ += self.num_motors
        if ENABLE_VALID_AMAT:
            if USE_NORMALIZED_ALLOCATION_MATRIX:
                if ENABLE_PINVERSE:
                    self.observations_tensor[:, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_normalized_pinv.view(-1, self.num_motors * 6) # self.normalized_allocation_matrix.view(-1, 24)
                else:
                    self.observations_tensor[:, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_normalized.view(-1, self.num_motors * 6) # self.normalized_allocation_matrix.view(-1, 24)
            else:
                if ENABLE_PINVERSE:
                    self.observations_tensor[:, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_pinverse.view(-1, self.num_motors * 6)
                else:
                    self.observations_tensor[:, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices.view(-1, self.num_motors * 6)
            self.index_ += self.num_motors * 6
        elif MOTOR_POSES:
            self.observations_tensor[:, self.index_:self.index_ + self.num_motors * 7] = valid_motor_poses.view(-1, self.num_motors * 7)
            self.index_ += self.num_motors * 7
        if MASS_AND_INERTIA_OBSERVATIONS:
            self.observations_tensor[:, self.index_] = valid_robot_mass[:, 0]  # mass
            self.observations_tensor[:, self.index_+1:self.index_+10] = valid_robot_inertias.view(-1, 9)  # inertia tensor
            self.index_ += 10
        if FORCE_AND_TIME_CONSTANTS:
            self.observations_tensor[:, self.index_ + self.num_motors:self.index_+2*self.num_motors] = valid_motor_force_constants[:,:,0]  # motor force constants
            self.observations_tensor[:, self.index_:self.index_ + self.num_motors] = valid_motor_time_constants[:, :, 0]  # motor time constants
            self.index_ += 2*self.num_motors

        print(valid_robot_inertias[5])

        
        return
    

    def init_tensors_from_dict_v2(self, loaded_configs=None, global_tensor_dict=None, mask=None):
        if global_tensor_dict is None:
            global_tensor_dict = {}
        if loaded_configs is None:
            raise ValueError("Config is None. Needs robot information.")

        loaded_configs["num_robots"] = int(loaded_configs["num_robots"])


        valid_robot_mass = torch.tensor(loaded_configs["robot_mass"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        #print("valid_robot_mass shape: ", valid_robot_mass.shape, torch.tensor(loaded_configs["robot_mass"]).shape, torch.tensor(loaded_configs["robot_mass"]).repeat_interleave(self.repeat_configs, dim=0).shape)
        valid_robot_inertias = torch.tensor(loaded_configs["robot_inertia"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_robot_inertias_inverse = torch.tensor(loaded_configs["robot_inertia_inverse"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_allocation_matrices = torch.tensor(loaded_configs["allocation_matrix"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_allocation_matrices_pinverse = torch.tensor(loaded_configs["allocation_matrix_inverse"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_max_thrusts = torch.tensor(loaded_configs["motor_max_thrusts"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_min_thrusts = torch.tensor(loaded_configs["motor_min_thrusts"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_time_constants = torch.tensor(loaded_configs["motor_time_constants"]).unsqueeze(-1).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_force_constants = torch.tensor(loaded_configs["motor_force_constants"]).unsqueeze(-1).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_directions = torch.tensor(loaded_configs["motor_directions"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_poses = torch.tensor(loaded_configs["motor_poses"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_torque_constants = torch.tensor(loaded_configs["motor_torque_constants"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_masses = torch.tensor(loaded_configs["motor_mass"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]
        valid_motor_inertias = torch.tensor(loaded_configs["motor_inertia"]).repeat_interleave(self.repeat_configs, dim=0)[:NUM_ENVS]

        
        
        if self.randomize_descriptor is not None:
            #uniform 0.1 mass, 
            mass_rel_std = self.randomize_descriptor['mass_rel_std']   # 2%
            inertia_rel_std = self.randomize_descriptor['inertia_rel_std'] # 5% (tune)

            m_obs = self.rel_noise(valid_robot_mass, mass_rel_std, clamp_min=1e-4)         # (B,1) or (B,)
            J_obs = self.rel_noise(valid_robot_inertias, inertia_rel_std)                   # (B,3,3)
            true_robot_mass = m_obs
            true_robot_inertias = J_obs

            # If your inertia is guaranteed SPD, inverse is fine; otherwise you may need SPD projection.
            Jinv_obs = torch.linalg.inv(J_obs)  # (B,3,3)
            true_robot_inertias_inverse = Jinv_obs
            true_motor_torque_constants = self.rel_noise(valid_motor_torque_constants, self.randomize_descriptor['torque_randomize_range'])  # (B,M,1)
            true_motor_force_constants = self.rel_noise(valid_motor_force_constants, self.randomize_descriptor['force_randomize_range'])  # (B,M,1)
            true_motor_time_constants = self.rel_noise(valid_motor_time_constants, self.randomize_descriptor['time_randomize_range'])  # (B,M,1)

            # --- compute noisy normalized allocation matrix using CLEAN allocation_matrix_tensor ---
            #A_clean = self.allocation_matrix_tensor[mask]  # (B,6,M) or whatever your shape is
            A = self.get_randomized_allocation_matrix_tensor(valid_motor_poses, valid_motor_directions, true_motor_torque_constants,
                                                             self.randomize_descriptor['motor_pos_randomize_range'], self.randomize_descriptor['torque_randomize_range'], 
                                                             self.randomize_descriptor['motor_orientation_randomize_range']).to(self.device)  # (B,6,M)            
            true_allocation_matrices = A


        else:
            true_robot_mass = valid_robot_mass
            true_robot_inertias = valid_robot_inertias
            true_robot_inertias_inverse = valid_robot_inertias_inverse
            true_allocation_matrices = valid_allocation_matrices
            true_motor_force_constants = valid_motor_force_constants
            true_motor_time_constants = valid_motor_time_constants
            true_motor_torque_constants = valid_motor_torque_constants
        
        
        
        self.mass_tensor[mask] = true_robot_mass[:].to(self.device)

        self.inertia_tensor[mask] = true_robot_inertias[:].to(self.device)
        self.inertia_inverse_tensor[mask] = true_robot_inertias_inverse[:].to(self.device)

        self.allocation_matrix_tensor[mask] = true_allocation_matrices[:].to(self.device)
        self.motor_direction_tensor[mask] = valid_motor_directions[:].to(self.device)

        self.allocation_matrix_tensor_pinv[mask] = torch.linalg.pinv(self.allocation_matrix_tensor[mask])


        self.normalized_allocation_matrix[mask] = self.get_normalized_allocation_matrix_tensor(
            self.allocation_matrix_tensor[mask], self.mass_tensor[mask], self.inertia_tensor[mask], self.inertia_inverse_tensor[mask]
        )

        
        
        valid_allocation_matrices_normalized = self.get_normalized_allocation_matrix_tensor(
            valid_allocation_matrices, valid_robot_mass, valid_robot_inertias, valid_robot_inertias_inverse
        )
        valid_allocation_matrices_normalized_pinv = torch.linalg.pinv(valid_allocation_matrices_normalized)
        self.normalized_allocation_matrix_pinverse[mask] = torch.linalg.pinv(self.normalized_allocation_matrix[mask])
        
        # # hardcoded values for sanity check
        self.motor_force_constant_tensor[mask] = true_motor_force_constants.to(self.device)
        self.motor_time_constant_tensor[mask] = true_motor_time_constants.to(self.device)
        

            
        self.index_ = 13
        if USE_MOTOR_ANGVEL:
            self.observations_tensor[mask, self.index_:self.index_+self.num_motors] = torch.zeros(mask.sum(), self.num_motors, device="cuda:0")  # motor angvels
            self.index_ += self.num_motors
        if ENABLE_VALID_AMAT:
            if USE_NORMALIZED_ALLOCATION_MATRIX:
                if ENABLE_PINVERSE:
                    self.observations_tensor[mask, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_normalized_pinv.view(-1, self.num_motors * 6).to(self.device) # self.normalized_allocation_matrix.view(-1, 24)
                else:
                    self.observations_tensor[mask, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_normalized.view(-1, self.num_motors * 6).to(self.device) # self.normalized_allocation_matrix.view(-1, 24)
            else:
                if ENABLE_PINVERSE:
                    self.observations_tensor[mask, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices_pinverse.view(-1, self.num_motors * 6).to(self.device)
                else:
                    self.observations_tensor[mask, self.index_:self.index_ + self.num_motors * 6] = valid_allocation_matrices.view(-1, self.num_motors * 6).to(self.device)
            self.index_ += self.num_motors * 6
        elif MOTOR_POSES:
            self.observations_tensor[mask, self.index_:self.index_ + self.num_motors * 7] = valid_motor_poses.view(-1, self.num_motors * 7).to(self.device)
            self.index_ += self.num_motors * 7
        else:
            self.observations_tensor[:, self.index_:] = 1.0
        if MASS_AND_INERTIA_OBSERVATIONS:
            self.observations_tensor[mask, self.index_] = valid_robot_mass[:, 0].to(self.device)  # mass
            self.observations_tensor[mask, self.index_+1:self.index_+10] = valid_robot_inertias.view(-1, 9).to(self.device)  # inertia tensor
            self.index_ += 10
        if FORCE_AND_TIME_CONSTANTS:
            self.observations_tensor[mask, self.index_ + self.num_motors:self.index_+2*self.num_motors] = valid_motor_force_constants[:,:,0].to(self.device)  # motor force constants
            self.observations_tensor[mask, self.index_:self.index_ + self.num_motors] = valid_motor_time_constants[:, :, 0].to(self.device)  # motor time constants
            self.index_ += 2*self.num_motors
        

        return


    def rel_noise(self, x, rel_std, clamp_min=0.0):
        if rel_std <= 0:
            return x
        y = x * (1.0 + (2*torch.rand_like(x) -1) * rel_std)
        if clamp_min is not None:
            y = y.clamp_min(clamp_min)
        return y

    def get_return_tuple(self):
        
        target_pos = torch.zeros_like(self.pos, device=self.device)
        target_vel = torch.zeros_like(self.vel, device=self.device)
        if self.follow_trajectory:
            if self.langevin:
                traj_pos, traj_vel, traj_pos_raw, traj_vel_raw = self.langevin_trajectories(self.traj_pos[self.traj==1], self.traj_vel[self.traj==1], 
                                                                                        self.traj_pos_raw[self.traj==1], self.traj_vel_raw[self.traj==1])
                self.traj_pos_raw[self.traj==1] = traj_pos_raw
                self.traj_vel_raw[self.traj==1] = traj_vel_raw
            elif self.lissajous:
                traj_pos, traj_vel = self.lissajous_traj['pos'][:,self.lis_time], self.lissajous_traj['vel'][:,self.lis_time]
                self.lis_time = (self.lis_time + 1) % self.lissajous_traj['pos'].shape[1]
            self.traj_pos[self.traj==1] = traj_pos
            self.traj_vel[self.traj==1] = traj_vel
            target_pos[self.traj==1] = traj_pos
            target_vel[self.traj==1] = traj_vel

        position_err = target_pos-self.root_state_tensor[:, 0:3]
        quat = self.root_state_tensor[:, 3:7]
        vel_err = target_vel-self.root_state_tensor[:, 7:10]
        angvel_err = self.root_state_tensor[:, 10:13]

        pos_error_world_frame = position_err
        vel_error_body_frame = quat_rotate_inverse(quat, vel_err)
        angvel_error_body_frame = quat_rotate_inverse(quat, angvel_err)

        if not self.noisy:
            self.observations_tensor[:, 0:3] = pos_error_world_frame
            self.observations_tensor[:, 3:7] = quat #self.quat_old #quat
            self.observations_tensor[:, 7:10] = vel_error_body_frame #self.vel_error_body_frame_old#vel_error_body_frame
            self.observations_tensor[:, 10:13] = angvel_error_body_frame #self.angvel_error_body_frame_old #
        else:
            sim_with_noise = 1.0 
            pos_noise = torch.normal(mean=torch.zeros_like(position_err), std=self.state_noise['pos']) * sim_with_noise
            obs_pos_noisy = position_err + pos_noise
            
            or_noise = torch.normal(mean=torch.zeros_like(quat[:,:3]), std=self.state_noise['ori']) * sim_with_noise
            or_quat = quat[:,[3, 0, 1, 2]]
            or_euler = matrix_to_euler_angles(quaternion_to_matrix(or_quat), "ZYX")[:, [2, 1, 0]]
            obs_or_euler_noisy = or_euler + or_noise
            lin_vel_noise = torch.normal(mean=torch.zeros_like(vel_error_body_frame), std=self.state_noise['linvel']) * sim_with_noise
            obs_linvel_noisy = vel_error_body_frame + lin_vel_noise
            
            ang_vel_noise = torch.normal(mean=torch.zeros_like(angvel_error_body_frame), std=self.state_noise['angvel']) * sim_with_noise #0.08
            ang_vel_noisy = self.angvel_error_body_frame_old + ang_vel_noise
            if self.delay_noise or self.enable_obs_lpf:
                not_initialized = ~self.obs_lpf_initialized

                # Initialize filter state from first noisy measurement
                if torch.any(not_initialized):
                    self.obs_pos_lpf[not_initialized] = obs_pos_noisy[not_initialized]
                    self.obs_euler_lpf[not_initialized] = obs_or_euler_noisy[not_initialized]
                    self.obs_linvel_lpf[not_initialized] = obs_linvel_noisy[not_initialized]
                    self.obs_angvel_lpf[not_initialized] = ang_vel_noisy[not_initialized]
                    self.obs_lpf_initialized[not_initialized] = True
            if self.delay_noise:
                # With probability p, don't update (use stale observation)
                update_mask = (torch.rand(self.num_envs, device=self.device) > self.delay_prob)  # (num_envs,)
                
                update_mask_3 = update_mask.unsqueeze(1).expand(-1, 3)
                update_mask_4 = update_mask.unsqueeze(1).expand(-1, 4)
                update_mask_euler = update_mask.unsqueeze(1).expand(-1, 3)

                obs_pos_noisy     = torch.where(update_mask_3,     obs_pos_noisy,     self.obs_pos_lpf)
                obs_or_euler_noisy = torch.where(update_mask_euler, obs_or_euler_noisy, self.obs_euler_lpf)
                obs_linvel_noisy  = torch.where(update_mask_3,     obs_linvel_noisy,  self.obs_linvel_lpf)
                ang_vel_noisy     = torch.where(update_mask_3,     ang_vel_noisy,     self.obs_angvel_lpf)

                # Store current (possibly stale) as the buffer for next step
                self.obs_pos_lpf      = obs_pos_noisy
                self.obs_euler_lpf    = obs_or_euler_noisy
                self.obs_linvel_lpf   = obs_linvel_noisy
                self.obs_angvel_lpf   = ang_vel_noisy
            
            if self.enable_obs_lpf:
                # First-order LPF
                self.obs_pos_lpf = (
                    self.obs_lpf_alpha_pos * obs_pos_noisy
                    + (1.0 - self.obs_lpf_alpha_pos) * self.obs_pos_lpf
                )

                self.obs_euler_lpf = (
                    self.obs_lpf_alpha_ori * obs_or_euler_noisy
                    + (1.0 - self.obs_lpf_alpha_ori) * self.obs_euler_lpf
                )

                self.obs_linvel_lpf = (
                    self.obs_lpf_alpha_linvel * obs_linvel_noisy
                    + (1.0 - self.obs_lpf_alpha_linvel) * self.obs_linvel_lpf
                )

                self.obs_angvel_lpf = (
                    self.obs_lpf_alpha_angvel * ang_vel_noisy
                    + (1.0 - self.obs_lpf_alpha_angvel) * self.obs_angvel_lpf
                )

                # Convert filtered Euler back to quaternion
                noisy_mat = euler_angles_to_matrix(self.obs_euler_lpf[:, [2, 1, 0]], "ZYX")
                noisy_quat = matrix_to_quaternion(noisy_mat)  # [w, x, y, z]

                self.observations_tensor[:, 0:3] = self.obs_pos_lpf
                self.observations_tensor[:, 3:7] = noisy_quat[:, [1, 2, 3, 0]]  # [x, y, z, w]
                self.observations_tensor[:, 7:10] = self.obs_linvel_lpf
                self.observations_tensor[:, 10:13] = self.obs_angvel_lpf

            else:
                noisy_mat = euler_angles_to_matrix(obs_or_euler_noisy[:, [2, 1, 0]], "ZYX")
                noisy_quat = matrix_to_quaternion(noisy_mat)  # [w, x, y, z]

                self.observations_tensor[:, 0:3] = obs_pos_noisy
                self.observations_tensor[:, 3:7] = noisy_quat[:, [1, 2, 3, 0]]
                self.observations_tensor[:, 7:10] = obs_linvel_noisy
                self.observations_tensor[:, 10:13] = ang_vel_noisy



       
        self.angvel_error_body_frame_old[:] = angvel_error_body_frame
        if USE_MOTOR_ANGVEL:
            self.observations_tensor[:, 13:13+self.num_motors] = self.motor_rpm_tensor.view(-1, self.num_motors)  # motor angvels
        

        if self.train_individual:
            self.obs_dict = {"obs": self.observations_tensor, "emb_ids": self.emb_id}
        else:
            self.obs_dict = {"obs": self.observations_tensor}
        self.info["crash_dict"] = self.crash_dict

        self.difficulty = 1 - (self.reward_mean - self.reward_mean.min()) / (self.reward_mean.max() - self.reward_mean.min())
        if torch.rand(1).item() < 0.001:
            print("samples per cfg:", self.samples_per_cfg.max(), self.samples_per_cfg.min(), self.samples_per_cfg.mean(), self.samples_per_cfg.sum()/2048)
            print("Reward mean: ", self.reward_mean.mean().item(), self.reward_mean.min().item(), self.reward_mean.max().item())
        return self.obs_dict, self.rewards, self.crashes.to(torch.bool), self.truncations, self.info

    
    

    def langevin_trajectories(self, pos, vel, pos_raw, vel_raw, dt=0.01, gamma=1.0, omega=2.0, sigma=0.5, alpha=0.01,
        dim=3,  seed=None,):
        """
        Simulate a batch of Langevin trajectories in parallel.

        Args:
            batch_size: number of parallel trajectories
            steps: number of timesteps
            dt: integration step
            gamma: damping coefficient
            omega: harmonic frequency
            sigma: noise strength
            alpha: smoothing factor
            dim: spatial dimension
            seed: RNG seed
        """
        rng = np.random.default_rng(seed)
        batch_size = pos_raw.shape[0]

        sqrt_dt = np.sqrt(dt)

        # previous state (safe at t=0 → zeros)
        x_prev      = pos_raw
        v_prev      = vel_raw
        v_smooth_prev = vel
        x_smooth_prev = pos

        # Gaussian noise for each trajectory and dimension
        dW = torch.tensor(sqrt_dt * rng.normal(size=(batch_size, dim)), device=self.device, dtype=pos.dtype)   

        # Euler–Maruyama update
        v_next = v_prev + (-gamma * v_prev - omega**2 * x_prev) * dt + sigma * dW
        x_next = x_prev + v_next * dt

        pos_raw = x_next
        vel_raw = v_next

        # Exponential smoothing
        v_smooth = alpha * v_next + (1 - alpha) * v_smooth_prev
        x_smooth = x_smooth_prev + v_smooth * dt

        vel = v_smooth
        pos = x_smooth

        return pos, vel, pos_raw, vel_raw


    def lissajous_figure_eight_torch(self, T=20.0, dt=0.01, Ax_range=(1.0, 1.0), Ay_range=(0.5, 0.5), fx_range=(0.1, 0.3),  fy=0.3, delta=torch.pi/2, center=(0.0, 0.0),
        batch_size=1, same = True):
        """
        Generate a batch of Lissajous figure-eight trajectories with random Ax, Ay, fx.
        """
        # Time vector
        t = torch.arange(0.0, T, dt, device=self.device)   # (T_steps,)
        T_steps = t.shape[0]
        print(batch_size)
        # Sample parameters per batch
        if same:
            Ax = (Ax_range[0] + 0.5*(Ax_range[1]-Ax_range[0]) * torch.ones((batch_size, 1), device=self.device))
            Ay = (Ay_range[0] + 0.5*(Ay_range[1]-Ay_range[0]) * torch.ones((batch_size, 1), device=self.device))
            fx = (fx_range[0] + 0.0*(fx_range[1]-fx_range[0]) * torch.ones((batch_size, 1), device=self.device))
        else:
            Ax = (Ax_range[0] + (Ax_range[1]-Ax_range[0]) * torch.rand((batch_size, 1), device=self.device))
            Ay = (Ay_range[0] + (Ay_range[1]-Ay_range[0]) * torch.rand((batch_size, 1), device=self.device))
            fx = (fx_range[0] + (fx_range[1]-fx_range[0]) * torch.rand((batch_size, 1), device=self.device))
        fy = 2*fx  # fix fy to be 2*fx for figure-eight
        delta = torch.as_tensor(delta, device=self.device).view(-1, 1).expand(batch_size, T_steps)
        cx = torch.as_tensor(center[0], device=self.device).view(-1, 1).expand(batch_size, T_steps)
        cy = torch.as_tensor(center[1], device=self.device).view(-1, 1).expand(batch_size, T_steps)

        # Time expanded for broadcasting [1, T_steps]
        t_exp = t.unsqueeze(0).expand(batch_size, T_steps)

        wx, wy = 2*torch.pi*fx, 2*torch.pi*fy

        # Position
        x = cx + Ax * torch.sin(wx * t_exp)
        y = cy + Ay * torch.cos(wy * t_exp + delta)
        z = torch.zeros_like(x) * 1.0  # constant height of 1.0
        # Velocity
        vx = Ax * wx * torch.cos(wx * t_exp)
        vy = -Ay * wy * torch.sin(wy * t_exp + delta)
        vz = torch.zeros_like(vx)

        # Acceleration
        ax = -Ax * (wx**2) * torch.sin(wx * t_exp)
        ay = -Ay * (wy**2) * torch.cos(wy * t_exp + delta)
        az = torch.zeros_like(ax)

        traj = {
            "t": t,  # (T_steps,)
            "pos": torch.stack([x, y, z], dim=-1),  # (batch, T_steps, 3)
            "vel": torch.stack([vx, vy, vz], dim=-1),
            "acc": torch.stack([ax, ay, az], dim=-1),
        }
        self.lissajous_traj = traj






