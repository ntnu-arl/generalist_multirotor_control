from generalist_multirotor_control.simulator.multirotor_dynamics_warp_kernels import (
    euler_quad_dynamics_update,
    rk4_quad_dynamics_update,
)

import numpy as np

from generalist_multirotor_control.utils.math_utils import quat_from_euler_xyz_tensor


import torch
import warp as wp



class MultirotorDynamicsEnv:
    def __init__(self, config, device="cuda:0"):
        seed = 42

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)



    def init_tensors(self, num_envs=2048, global_tensor_dict=None):
        return

    def capture_graphs(self):

        wp.capture_begin()
        wp.launch(
            kernel=euler_quad_dynamics_update,
            dim=self.num_envs,  # This sets the number of parallel executions
            inputs=[
                self.pos_wp,
                self.quat_wp,
                self.vel_wp,
                self.angvel_wp,
                self.motor_rpm_wp,
                self.motor_setpoint_wp,
                self.motor_direction_wp,
                self.motor_force_constant_wp,
                self.motor_time_constant_wp,
                self.mass_wp,
                self.inertia_wp,
                self.inertia_inverse_wp,
                self.gravity_wp,
                self.amat_wp,
                self.dt,
            ],
            device=self.device,
        )
        self.euler_graph = wp.capture_end()

        wp.capture_begin()
        wp.launch(
            kernel=rk4_quad_dynamics_update,
            dim=self.num_envs,  # This sets the number of parallel executions
            inputs=[
                self.pos_wp,
                self.quat_wp,
                self.vel_wp,
                self.angvel_wp,
                self.motor_rpm_wp,
                self.motor_setpoint_wp,
                self.motor_direction_wp,
                self.motor_force_constant_wp,
                self.motor_time_constant_wp,
                self.mass_wp,
                self.inertia_wp,
                self.inertia_inverse_wp,
                self.gravity_wp,
                self.amat_wp,
                self.dt,
            ],
            device=self.device,
        )
        self.rk4_graph = wp.capture_end()

        return

    def step(self, actions, metric_force=False):
        if not metric_force:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            scaled_clamped_actions = ((clamped_actions + 1) * 0.5) * 2.0
        else:
            scaled_clamped_actions = actions
        setpoint_rpm = torch.sqrt(
            scaled_clamped_actions / self.motor_force_constant_tensor.squeeze(-1)
        )
        if self.step_counter%1==0:
            self.prev_actions[:] = self.actions[:]
        self.actions[:] = scaled_clamped_actions[:]
        self.motor_setpoint_tensor[:] = setpoint_rpm.unsqueeze(-1)
        wp.capture_launch(self.rk4_graph)


        self.rewards[:], self.crashes[:], self.crash_dict, prev_angvel = compute_rewards(
            self.pos, self.quat, self.vel, self.angvel, self.actions, self.prev_actions, self.prev_angvel
        )

        if self.step_counter%1==0:
            self.prev_angvel = prev_angvel


        alpha = 0.001
        self.update_reward_mean(alpha)
        self.truncations = torch.where(
            self.counter_tensor > self.max_episode_steps,
            torch.ones_like(self.counter_tensor, dtype=torch.bool),
            torch.zeros_like(self.counter_tensor, dtype=torch.bool),
        )
        # index of the ids that are either truncated or terminated
        self.crash_dict["timeout"] = self.truncations
        reset_ids = torch.where(self.crashes + self.truncations > 0)[0]
        self.reset_idx(reset_ids)
        self.quat[:] = self.quat*(torch.sign(self.quat[:, 3]).unsqueeze(-1))
        
        self.counter_tensor += 1
        self.step_counter += 1
        if self.step_counter % 32 == 0:
            self.train_epochs += 1 
            if self.train_epochs % 50 == 0:
                print('\n\nepoch changed', self.train_epochs, "\n\n")
        return self.get_return_tuple()

    def get_return_tuple(self):
        self.obs_dict = {"observations": self.root_state_tensor}
        return self.obs_dict, self.rewards, self.crashes.to(torch.bool), self.truncations, self.info

    def reset(self):
        self.reset_idx(torch.arange(self.root_state_tensor.size(0)))
        self.crashes = torch.zeros(self.root_state_tensor.size(0), device=self.device)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        # if env_ids are empty, return
        if env_ids.size(0) == 0:
            return
        if self.reset_counter % 500 == 0:
            print("\n\n\n average episode length: ", self.counter_tensor[env_ids].float().mean().item(), "\n\n\n")

        self.reset_counter += 1
        self.motor_rpm_tensor[env_ids, :] = 0.0
        if not self.follow_trajectory:
            p = 0.3
            if not self.one_point:
                self.pos[env_ids, :] = 0.5*(-1.0 + 2.0 * torch.rand(self.pos[env_ids].shape, device=self.device))
                self.vel[env_ids, :] = -0.5 + 1.0 * torch.rand(self.vel[env_ids].shape, device=self.device)  
                self.angvel[env_ids, :] = -1 + 2 * torch.rand(
                    self.angvel[env_ids].shape, device=self.device
                )
                # self.angvel[env_ids, :] = 0.0
                # self.vel[env_ids, :] = 0.0
            else:
                self.pos[env_ids, :] = 0.0*torch.ones(self.pos[env_ids].shape, device=self.device)
                self.pos[env_ids, 0] = 0.2
                self.vel[env_ids, :] = 0.0 * torch.ones(self.vel[env_ids].shape, device=self.device)  
                self.angvel[env_ids, :] = 0.0* torch.ones(
                    self.angvel[env_ids].shape, device=self.device
                )
                p = 0.0
            
            # --- choose which envs go to exact-zero reset ---
            # mask shape: (num_envs_reset,)
            zero_mask = (torch.rand((len(env_ids),), device=self.device) < p)

            # expand to (num_envs_reset, 3)
            zero_mask3 = zero_mask.unsqueeze(-1)

            # --- apply: with prob p -> zeros, else -> defaults ---
            self.pos[env_ids, :]    = torch.where(zero_mask3, torch.zeros_like(self.pos[env_ids, :]),    self.pos[env_ids, :])
            self.vel[env_ids, :]    = torch.where(zero_mask3, torch.zeros_like(self.vel[env_ids, :]),    self.vel[env_ids, :])
            self.angvel[env_ids, :] = torch.where(zero_mask3, torch.zeros_like(self.angvel[env_ids, :]), self.angvel[env_ids, :])
        else:
            self.pos[env_ids, :] = self.traj_pos[env_ids, :] + 0.2*torch.rand(self.pos[env_ids].shape, device=self.device)
            self.vel[env_ids, :] = self.traj_vel[env_ids, :] 
            self.angvel[env_ids, :] = 0
        if False:
            min_init_quat = torch.tensor([-np.pi/6,-np.pi/6,-np.pi/2, 1.0], device=self.device).unsqueeze(0).expand(len(env_ids), -1)
            max_init_quat = torch.tensor([np.pi/6,np.pi/6,np.pi/2, 1.0], device=self.device).unsqueeze(0).expand(len(env_ids), -1)
            random_quat = min_init_quat + (max_init_quat - min_init_quat) * torch.rand((len(env_ids), 4), device=self.device)
            rand = quat_from_euler_xyz_tensor(random_quat)
            self.quat[env_ids, :] = rand
            if self.one_point:
                self.quat[env_ids, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(len(env_ids), -1)

        self.quat[env_ids, :] = 0.0
        self.quat[env_ids, 3] = 1.0

        self.counter_tensor[env_ids] = 0
    
    #For each configuration that appeared in this batch, compute its average reward 
    # and softly update the stored mean reward using exponential smoothing.
    def update_reward_mean(self, alpha):
        config_ids = self.rw2cfg_permutation
        r = self.rewards.detach()
        x = self.reward_mean

        reward_sum = torch.zeros_like(x).scatter_add(0, config_ids, r)
        count = torch.zeros_like(x).scatter_add(0, config_ids, torch.ones_like(r))
        mean_reward = reward_sum / (count + 1e-8)

        mask = count > 0
        x[mask] = (1 - alpha) * x[mask] + alpha * mean_reward[mask]

@torch.jit.script
def exp_rew_func(x: torch.Tensor, a: float, b: float):
    return a * torch.exp(-b * x * x)

@torch.jit.script
def exp_penalty_func(x: torch.Tensor, a: float, b: float):
    return a * (torch.exp(-b * x * x) - 1)


@torch.jit.script
def ssa(x: torch.Tensor):
    # input is roll, pitch, yaw, return values between -pi/2 and pi/2
    return torch.minimum(torch.abs(x), torch.abs(x - 2.0 * 3.14159))


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor):
    qw = quat[:, 3]
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(2.0 * (qw * qy - qz * qx))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack([roll, pitch, yaw], dim=1)


def compute_rewards(pos, quat, vel, angvel, actions, prev_actions, prev_angvel):
    dist = torch.norm(pos, dim=1)
    pos_reward = exp_rew_func(dist, 3.0, 5.0) + exp_rew_func(dist, 6.0, 12.0) # 4, 10

    vel_norm = torch.norm(vel, dim=1)
    vel_reward = exp_rew_func(vel_norm, 2.0, 10.0) 

    angvel_norm = torch.norm(angvel, dim=1)
    angvel_reward = exp_rew_func(angvel_norm, 1.0, 0.5) #1, 0.5

    ang_acc = torch.norm(angvel - prev_angvel, dim=1) 
    ang_acc_reward = exp_rew_func(ang_acc, 1.0, 0.5)

    angles = ssa(euler_xyz_from_quat(quat))
    yaw_reward = exp_rew_func(angles[:, 2], 0.75, 6.0)

    action_difference_penalty = - 0.1 * torch.norm((actions - prev_actions), dim=1)
    action_scaling = 0.4*9.81*0.5
    action_magnitude_penalty = - 0.5 * torch.norm((actions+1)*0.5*action_scaling - action_scaling/3, dim=1) #0.1
    
    proximity = torch.exp(-dist * 5.0)
                                  
    total_reward = (
        pos_reward
        + vel_reward
        + angvel_reward
        + ang_acc_reward
        + proximity * (4*vel_reward + 16*angvel_reward + yaw_reward) #4 8 1
        + action_difference_penalty

        )

    crashes = torch.where(
        dist > 4.0,
        torch.ones_like(dist, dtype=torch.bool),
        torch.zeros_like(dist, dtype=torch.bool),
    )
    crashes[:] = torch.where(angvel_norm > 20.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(vel_norm > 20.0, torch.ones_like(crashes), crashes)

    crash_due_to_pos = torch.where(dist > 4.0, torch.ones_like(crashes), torch.zeros_like(crashes))
    crash_due_to_angvel = torch.where(
        angvel_norm > 20.0, torch.ones_like(crashes), torch.zeros_like(crashes)
    )
    crash_due_to_vel = torch.where(
        vel_norm > 20.0, torch.ones_like(crashes), torch.zeros_like(crashes)
    )

    # cause dict
    crash_dict = {"angvel": crash_due_to_angvel, "vel": crash_due_to_vel, "pos": crash_due_to_pos,
                  'action_dif': action_magnitude_penalty, 'pos_rew': pos_reward,
                  'ang_acc': ang_acc_reward, 'angvel_rew': angvel_reward, 'vel_rew': vel_reward}
    total_reward[:] = torch.where(crashes, -200.0, total_reward) #200
    return total_reward, crashes, crash_dict, angvel.clone()



def compute_rewards_sim2real(pos, quat, vel, angvel, actions, prev_actions, prev_angvel):
    dist = torch.norm(pos, dim=1)
    pos_reward = exp_rew_func(dist, 3.0, 3.0) + exp_rew_func(dist, 4.0, 6.0) # 3, 5, 4, 8 | 3 5 4 6 |

    vel_norm = torch.norm(vel, dim=1)
    vel_reward = exp_rew_func(vel_norm, 2.0, 5.0) # 2 10

    angvel_norm = torch.norm(angvel, dim=1)
    angvel_reward = exp_rew_func(angvel_norm, 1.0, 0.3) #1, 0.3

    ang_acc = torch.norm(angvel - prev_angvel, dim=1) 
    ang_acc_reward = exp_rew_func(ang_acc, 1.0, 0.5)

    angles = ssa(euler_xyz_from_quat(quat))
    roll_reward = exp_rew_func(angles[:, 0], 0.6, 6.0)
    pitch_reward = exp_rew_func(angles[:, 1], 0.6, 6.0)
    yaw_reward = exp_rew_func(angles[:, 2], 0.75, 6.0)

    action_difference_penalty = - 0.5 * torch.norm((actions - prev_actions), dim=1) 

    total_reward = (
        pos_reward
        + vel_reward
        + angvel_reward
        + ang_acc_reward
        + (1/7) *pos_reward * (1*vel_reward + 8*angvel_reward + yaw_reward) 
        + action_difference_penalty
        )

    crashes = torch.where(
        dist > 4.0,
        torch.ones_like(dist, dtype=torch.bool),
        torch.zeros_like(dist, dtype=torch.bool),
    )
    crashes[:] = torch.where(angvel_norm > 20.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(vel_norm > 20.0, torch.ones_like(crashes), crashes)

    crash_due_to_pos = torch.where(dist > 4.0, torch.ones_like(crashes), torch.zeros_like(crashes))
    crash_due_to_angvel = torch.where(
        angvel_norm > 20.0, torch.ones_like(crashes), torch.zeros_like(crashes)
    )
    crash_due_to_vel = torch.where(
        vel_norm > 20.0, torch.ones_like(crashes), torch.zeros_like(crashes)
    )

    # cause dict
    crash_dict = {"angvel": crash_due_to_angvel, "vel": crash_due_to_vel, "pos": crash_due_to_pos,
                  'action_dif': action_difference_penalty, 'pos_rew': pos_reward,
                  'ang_acc': action_difference_penalty, 'angvel_rew': angvel_reward, 'vel_rew': vel_reward}
    total_reward[:] = torch.where(crashes, -200.0, total_reward) 
    return total_reward, crashes, crash_dict, angvel.clone()


    