import random
import time
from generalist_multirotor_control.utils.math_utils import *
from torch.optim import Adam
import torch
import numpy as np

# @torch.jit.script
def sample_quaternion(rows, cols): 
    
    """
    Sample a random quaternion with the same shape as the input quaternion tensor.
    The input tensor is expected to have a shape where the last dimension is 4 (quaternion).
    """
    shape = (rows, cols, 4)
    u1 = torch.rand(shape[0], shape[1], dtype=torch.float32, device='cuda')
    u2 = torch.rand(shape[0], shape[1], dtype=torch.float32, device='cuda')
    u3 = torch.rand(shape[0], shape[1], dtype=torch.float32, device='cuda')
    
    qx = torch.sqrt(1.0 - u1) * torch.sin(2.0 * torch.pi * u2)
    qy = torch.sqrt(1.0 - u1) * torch.cos(2.0 * torch.pi * u2)
    qz = torch.sqrt(u1) * torch.sin(2.0 * torch.pi * u3)
    qw = torch.sqrt(u1) * torch.cos(2.0 * torch.pi * u3)
    q0 = qx
    q1 = qy
    q2 = qz
    q3 = qw
    return torch.stack([q0, q1, q2, q3], dim=-1)
    

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_rotate_multidim(q, v):
    shape = v.shape
    q_flat = q.view(-1, 4)
    v_flat = v.view(-1, 3)
    rotated_vec = quat_rotate(q_flat, v_flat)
    return rotated_vec.view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Both (...,4) in (x,y,z,w)
        x1,y1,z1,w1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
        x2,y2,z2,w2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
        x =  w1*x2 + x1*w2 + y1*z2 - z1*y2
        y =  w1*y2 - x1*z2 + y1*w2 + z1*x2
        z =  w1*z2 + x1*y2 - y1*x2 + z1*w2
        w =  w1*w2 - x1*x2 - y1*y2 - z1*z2
        return torch.stack((x,y,z,w), dim=-1)

@torch.jit.script
def position_from_spherical(range, azimuth, elevation):
    x = range * torch.cos(elevation) * torch.cos(azimuth)
    y = range * torch.cos(elevation) * torch.sin(azimuth)
    z = range * torch.sin(elevation)
    return torch.stack([x, y, z], dim=-1)

@torch.jit.script
def inertia_box_3d(mass, length, width, height):
    """
    Calculate the inertia tensor for a rectangular box with given dimensions.
    
    Args:
        mass (torch.Tensor): Mass of the box.
        length (torch.Tensor): Length of the box along the x-axis.
        width (torch.Tensor): Width of the box along the y-axis.
        height (torch.Tensor): Height of the box along the z-axis.
    
    Returns:
        torch.Tensor: Inertia tensor of shape (N, 3, 3).
    """
    mass = mass.squeeze(-1)
    Ixx = (1/12) * mass * (width**2 + height**2)
    Iyy = (1/12) * mass * (length**2 + height**2)
    Izz = (1/12) * mass * (length**2 + width**2)
    
    Ixy = torch.zeros_like(Ixx)
    Ixz = torch.zeros_like(Ixx)
    Iyz = torch.zeros_like(Ixx)

    I = torch.zeros(mass.shape[0], 3, 3, dtype=torch.float32, device=mass.device)
    # Fill the inertia tensor
    I[:, 0, 0] = Ixx
    I[:, 0, 1] = Ixy
    I[:, 0, 2] = Ixz
    I[:, 1, 0] = Ixy
    I[:, 1, 1] = Iyy
    I[:, 1, 2] = Iyz
    I[:, 2, 0] = Ixz
    I[:, 2, 1] = Iyz
    I[:, 2, 2] = Izz
    return I


def transform_inertia_tensor(mass: torch.Tensor, inertia_tensor: torch.Tensor, relative_pose: torch.Tensor) -> torch.Tensor:
    """
    Transform the inertia tensor from the local frame to the world frame using the relative pose.
    
    Args:
        mass (torch.Tensor): Mass of the object (N,).
        inertia_tensor (torch.Tensor): Inertia tensor of shape (N, 3, 3).
        relative_pose (torch.Tensor): Relative pose of shape (N, 7) in quaternion format (xyz + quaternion).
    
    Returns:
        torch.Tensor: Transformed inertia tensor of shape (N, 3, 3).
    """
    # Extract position and quaternion from the relative pose
    position = relative_pose[:, :3]  # Shape: (N, 3)
    quaternion = relative_pose[:, 3:]  # Shape: (N, 4)

    # Step 1: Rotate the inertia tensor using the quaternion
    rotation_matrix = quat_to_rotation_matrix(quaternion)  # Shape: (N, 3, 3)
    rotated_inertia = torch.bmm(torch.bmm(rotation_matrix, inertia_tensor), rotation_matrix.transpose(1, 2))  # Shape: (N, 3, 3)

    # Step 2: Apply the parallel axis theorem
    # Compute the outer product of the position vector
    position_outer = torch.bmm(position.unsqueeze(-1), position.unsqueeze(1))  # Shape: (N, 3, 3)
    position_squared = torch.sum(position ** 2, dim=-1).unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1, 1)

    if mass.dim() == 1:
        mass = mass.unsqueeze(-1)

    # Parallel axis theorem: I_new = I_rotated + m * (|r|^2 * I - r ⊗ r)
    parallel_axis_term = mass.unsqueeze(-1) * (position_squared * torch.eye(3, device=mass.device).unsqueeze(0) - position_outer)  # Shape: (N, 3, 3)
    transformed_inertia = rotated_inertia + parallel_axis_term  # Shape: (N, 3, 3)

    return transformed_inertia
    


@torch.jit.script
def inertia_sphere_3d(mass, radius):
    """
    Calculate the inertia tensor for a sphere with given mass and radius.
    
    Args:
        mass (torch.Tensor): Mass of the sphere.
        radius (torch.Tensor): Radius of the sphere.
    
    Returns:
        torch.Tensor: Inertia tensor of shape (N, 3, 3).
    """
    i = (2/5) * mass * radius**2
    I = torch.zeros(mass.shape[0], 3, 3, dtype=torch.float32, device=mass.device)
    I[:, 0, 0] = i
    I[:, 1, 1] = i
    I[:, 2, 2] = i
    return I

@torch.jit.script
def spherical_from_position(pos):
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    range = torch.norm(pos, dim=-1)
    azimuth = torch.atan2(y, x)
    elevation = torch.asin(z / range)
    return range, azimuth, elevation

def calculate_allocation_matrix(motor_poses, motor_torque_constants, motor_directions):
    num_robots = motor_poses.shape[0]
    num_motors = motor_poses.shape[1]
    allocation_matrix = torch.zeros((num_robots, 6, num_motors), dtype=torch.float32).cuda()

    motor_force_frame = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda().unsqueeze(0).expand(num_robots, num_motors, -1)
    motor_torque = motor_force_frame * motor_torque_constants.unsqueeze(-1)
    motor_pos = motor_poses[..., 0:3]
    motor_quat = motor_poses[..., 3:7]

    # Transform force and torque to robot frame
    motor_force_in_robot_frame = quat_rotate_multidim(motor_quat, motor_force_frame)
    motor_torque_in_robot_frame = quat_rotate_multidim(motor_quat, motor_torque*(-motor_directions.unsqueeze(-1)))

    # Calculate torque as r x F + tau
    torque = torch.cross(motor_pos, motor_force_in_robot_frame) + motor_torque_in_robot_frame

    # Fill the allocation matrix
    allocation_matrix[:, 0:3, :] = motor_force_in_robot_frame.transpose(1, 2)
    allocation_matrix[:, 3:6, :] = torque.transpose(1, 2)
    return allocation_matrix

def calculate_motor_directions(motor_positions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the motor directions based on their positions.
    The directions are assigned alternately to neighboring motors in 1 (CCW) or -1 (CW) based on their
    azimuthal position measured clockwise in the range 0 to 2pi from the robot front based on their x-y plane position
    
    Args:
        motor_positions (torch.Tensor): Shape (N, M, 3) where N is the number of robots and M is the number of motors.
    
    Returns:
        torch.Tensor: A tensor of shape (N, M) with values 1 or -1 indicating the motor direction.
    """
    azimuths = torch.atan2(motor_positions[:, :, 1], motor_positions[:, :, 0])  # Calculate azimuth angles
    azimuths = azimuths % (2 * torch.pi)  # Normalize angles to [0, 2pi]
    sort_indices = torch.argsort(azimuths, dim=1, descending=True)  # Sort azimuths in descending order
    

    # Create alternating directions tensor
    directions = torch.ones_like(sort_indices, dtype=torch.float32).cuda()
    directions[:, 1::2] = -1.0  # Alternate between 1 and -1

    # # Scatter directions based on sorted indices
    # motor_directions = torch.zeros_like(directions).cuda()
    # motor_directions.scatter_(1, sort_indices, directions)
    motor_directions = directions
    return motor_directions

def get_normalized_allocation_matrix_tensor(
        allocation_matrix_tensor, mass_tensor, inertia_inverse_tensor
    ):
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

class AirframeSamplerCfg:
    """
    Configuration class for the AirframeSampler, which defines the sampling
    parameters for generating drone designs.
    """
    def __init__(self, num_samples: int = 1000, planar = False, random = False, symmetric = False, wide = False, payload = True, motor_count: int = 4, device = 'cuda'):
        
        self.num_samples = num_samples
        self.motor_count = motor_count
        self.planar = planar
        self.random = random
        self.symmetric = symmetric
        self.wide = wide
        self.device = device

        # robot core configs
        self.robot_core_mass = torch.zeros(self.num_samples, 1, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_core_inertia = torch.zeros(self.num_samples, 3, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_core_pose = torch.zeros(self.num_samples, 7, dtype=torch.float32, requires_grad=False, device=self.device)

        # motor configs
        self.motor_poses = torch.zeros(self.num_samples, self.motor_count, 7, dtype=torch.float32, requires_grad=False, device=self.device)
        self.motor_min_thrusts = torch.zeros(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device)
        self.motor_max_thrusts = torch.ones(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device)
        self.motor_inertias = torch.zeros(self.num_samples, self.motor_count, 3, 3, dtype=torch.float32, requires_grad=False, device=self.device)

        self.motor_time_constants_min = torch.ones(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device) * 0.01 #0.01
        self.motor_time_constants_max = torch.ones(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device) * 0.05 #0.05

        self.motor_time_constants = torch_rand_float_tensor(
            lower= self.motor_time_constants_min,
            upper= self.motor_time_constants_max,
        )

        if (self.random or self.symmetric or self.planar) and not self.wide:
            self.motor_time_constants[:] = 0.047 * torch.ones_like(self.motor_time_constants)  # Set all motors to a specific value (example for DJI E305)


        # sampling parameters
        ones_mass = torch.ones(self.num_samples, 1, dtype=torch.float32, requires_grad=False, device=self.device)
        ones_motors = torch.ones(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device)
        if self.wide or (not (self.random or self.symmetric or self.planar)):
            self.robot_core_mass_min = 0.2 
            self.robot_core_mass_max = 1.3 
        else:
                self.robot_core_mass_min = 0.324
                self.robot_core_mass_max = 0.324 #0.501 # 1.25
        self.motor_mass_min = 0.013 #0.01 #0.02
        self.motor_mass_max = 0.013 #0.04
        self.robot_core_mass_min = self.robot_core_mass_min * ones_mass
        self.robot_core_mass_max = self.robot_core_mass_max * ones_mass
        self.motor_mass_min = self.motor_mass_min * ones_motors
        self.motor_mass_max = self.motor_mass_max * ones_motors
        self.motor_inertia = torch.zeros(self.num_samples, self.motor_count, 3, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.motor_inertia[:, :, 0, 0] = 2.0e-6
        self.motor_inertia[:, :, 1, 1] = 2.0e-6
        self.motor_inertia[:, :, 2, 2] = 2.0e-6
        
        self.motor_max_thrusts = 12.5 * self.motor_max_thrusts
        self.motor_min_thrusts = 0.0 * self.motor_min_thrusts
        self.payload = payload
        self.motor_force_constants =  torch_rand_float_tensor(
            lower= 0.00001 * ones_motors,
            upper= 0.00005 * ones_motors,
        )
        if (self.random or self.symmetric or self.planar) and not self.wide:
            self.motor_force_constants[:] = 0.00002308   #0.00001286412  

        self.motor_torque_constants = torch_rand_float_tensor(
            lower= 0.01000 * ones_motors,       
            upper= 0.05000 * ones_motors,
        )
        if (self.random or self.symmetric or self.planar) and not self.wide:
            self.motor_torque_constants[:] = 0.01  

        self.robot_core_mass = torch.zeros(self.num_samples, 1, dtype=torch.float32, requires_grad=False, device=self.device)
        self.motor_mass = torch.zeros(self.num_samples, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device)

        # derived values
        self.robot_mass = torch.zeros(self.num_samples, 1, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_com = torch.zeros(self.num_samples, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_inertia = torch.zeros(self.num_samples, 3, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.allocation_matrix = torch.zeros(self.num_samples, 6, self.motor_count, dtype=torch.float32, requires_grad=False, device=self.device)


    def __repr__(self):
        return f"AirframeSamplerCfg(num_samples={self.num_samples}, motor_count={self.motor_count})"

    def calculate_robot_inertia(self):
        """
        Calculate the robot inertia based on the motor poses and masses.
        
        Returns:
            torch.Tensor: The calculated inertia tensor of the robot.
        """
        
        # Get positions relative to robot COM
        motor_poses_com = self.motor_poses.clone()
        motor_poses_com[..., :3] -= self.robot_com.unsqueeze(1)
        
        # Assuming robot_core_pose is available (position and orientation of core relative to robot COM)
        # If core is at robot COM, then robot_core_pose would be [0, 0, 0, 0, 0, 0, 1]
        core_pose_com = self.robot_core_pose.clone()
        core_pose_com[..., :3] -= self.robot_com

        ## Payload com
        self.payload_poses_com = self.payload_poses.clone()

        # print ("payload poses shape:", self.payload_poses.shape)

        self.payload_poses_com[..., :3] -= self.robot_com
        
        
        self.motor_poses_com = motor_poses_com.clone()
        
        # Initialize robot inertia tensor
        robot_inertia = torch.zeros(self.num_samples, 3, 3, dtype=torch.float32, requires_grad=False, device=self.robot_core_inertia.device)


        robot_inertia[:] += transform_inertia_tensor(
            mass=self.robot_core_mass,
            inertia_tensor=self.robot_core_inertia,
            relative_pose=core_pose_com
        )

        robot_inertia[:] += transform_inertia_tensor(
            mass=self.payload_mass,
            inertia_tensor=self.payload_inertia,
            relative_pose=self.payload_poses_com
        )

        for i in range(self.motor_count):
            # Transform each motor inertia tensor to the robot frame
            motor_inertia_tensor = transform_inertia_tensor(
                mass=self.motor_mass[:, i],
                inertia_tensor=self.motor_inertia[:, i],
                relative_pose=self.motor_poses_com[:, i]
            )
            
            # Add the transformed motor inertia tensor to the robot inertia tensor
            robot_inertia += motor_inertia_tensor

        return robot_inertia

    
    def sampling_method_planar(self):
        min_range = 0.1371 #0.10
        max_range = 0.1371 #0.25
        range_ = torch_rand_float(lower=min_range, upper=max_range, shape=(self.num_samples, 1), device=self.device)
        r = 0 #30 #30
        step = 360//self.motor_count
        start = (360 - self.motor_count*r)//8 if self.motor_count==4 else 30
        azimuth_ranges = [(start+step*i, start+step*i+r) for i in range(self.motor_count)]
        # azimuth_ranges = [(15, 75), (105,165), (195,255), (285,345)]
        for i,(a0,a1) in enumerate(azimuth_ranges):
            self.motor_poses[:, i, 0:3] = position_from_spherical(
                range=torch_rand_float(min_range, max_range, (self.num_samples,1), device=self.device),
                azimuth=torch_rand_float(a0*torch.pi/180, a1*torch.pi/180, (self.num_samples,1), device=self.device),
                elevation=torch.zeros((self.num_samples,1), device=self.device)
            ).squeeze(1)
        identity_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device)
        self.motor_poses[:, :, 3:7] = identity_quat.view(1, 1, 4).expand(self.num_samples, self.motor_count, 4)
    
    def sampling_method_planar_wide(self):
        min_range = 0.10
        max_range = 0.25
        range_ = torch_rand_float(lower=min_range, upper=max_range, shape=(self.num_samples, 1), device=self.device)
        r = 45 #30
        step = 360//self.motor_count
        start = 0
        azimuth_ranges = [(start+step*i, start+step*i+r) for i in range(self.motor_count)]
        # azimuth_ranges = [(15, 75), (105,165), (195,255), (285,345)]
        for i,(a0,a1) in enumerate(azimuth_ranges):
            self.motor_poses[:, i, 0:3] = position_from_spherical(
                range=range_,
                azimuth=torch_rand_float(a0*torch.pi/180, a1*torch.pi/180, (self.num_samples,1), device=self.device),
                elevation=torch.zeros((self.num_samples,1), device=self.device)
            ).squeeze(1)
        identity_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device)
        self.motor_poses[:, :, 3:7] = identity_quat.view(1, 1, 4).expand(self.num_samples, self.motor_count, 4)
    


    def sampling_method_symmetric(self):

        self.motor_poses = torch.tensor([[0.106471, -0.061471, 0.04, 0.174294, -0.17863, -0.223234, 0.942274],
               [0, -0.122942, -0.03, -0.241845, 0.061628, -0.704416, 0.664463],
                [-0.106471, -0.061471, 0.04, -0.067551, -0.240258, -0.92765, 0.277811],
                [-0.106471, 0.061471, 0.04, 0.067551, -0.240258, 0.92765, 0.277811],
                [0, 0.122942, -0.03, 0.241845, 0.061628, 0.704416, 0.664463],
                [0.106471, 0.061471, 0.04, -0.174294, -0.17863, 0.223234, 0.942274]]).flip(dims=[0]).unsqueeze(0).expand(self.num_samples, -1, -1).to(self.device)
        # self.motor_poses = torch.tensor([[ 0.0869, -0.0869,  0.0400,  0.1743, -0.1786, -0.2232,  0.9423],
        # [-0.0318, -0.1188, -0.0300, -0.2418,  0.0616, -0.7044,  0.6645],
        # [-0.1188, -0.0318,  0.0400, -0.0676, -0.2403, -0.9276,  0.2778],
        # [-0.0869,  0.0869,  0.0400,  0.0676, -0.2403,  0.9276,  0.2778],
        # [ 0.0318,  0.1188, -0.0300,  0.2418,  0.0616,  0.7044,  0.6645],
        # [ 0.1188,  0.0318,  0.0400, -0.1743, -0.1786,  0.2232,  0.9423]]).flip(dims=[0]).unsqueeze(0).expand(self.num_samples, -1, -1).to(self.device)
    


    def sampling_method_random(self):
       
        self.motor_poses[:, :, 0:3] = torch.tensor([[ 0.09677586, -0.08848868, -0.0490325 ],
                            [ 0.01612816, -0.13240591,  0.04252714],
                            [-0.0937321, -0.0658898 , -0.05741112],
                            [-0.10844458,  0.06142187,  0.06377404],
                            [-0.01158604,  0.13887489,  0.01339886],
                            [ 0.12783,  0.037545,  0.02263044]]).flip(dims=[0]).repeat(self.num_samples,1,1).to(self.device)

        sampled_angles = torch.flip(torch.tensor([[  0.0000,  13.4209, 184.5375],
                                    [  0.0000,  21.8579, 151.9435],
                                    [  0.0000,   5.0737, 253.7144],
                                    [  0.0000,   9.8699, 216.3987],
                                    [  0.0000,  11.1912,  91.0281],
                                    [  0.0000,   5.1519, 234.2503]]), dims=[0])
        real_angles = sampled_angles.clone()
        sampled_angles = sampled_angles.repeat(self.num_samples,1,1).to(self.device)
        sampled_angles = torch.deg2rad(sampled_angles)
        self.motor_poses[:, :, 3:7] = quat_from_euler_xyz_tensor(sampled_angles)


    def sampling_method_all(self):
            min_range = 0.1 #0.15
            max_range = 0.3
            min_pitch = -torch.pi/4.0 
            max_pitch = torch.pi/4.0

            # sample motor positions in spherical, 4 motors at 90 deg offsets
            r = 65 if self.motor_count==4 else 45 #30
            step = 360//self.motor_count
            #start = (360 - self.motor_count*r)//8 if self.motor_count==4 else 15
            start = 0 #if self.motor_count==4 else 15
            azimuth_ranges = [(start+step*i, start+step*i+r) for i in range(self.motor_count)]
            # azimuth_ranges = [(15, 75), (105,165), (195,255), (285,345)]
            for i,(a0,a1) in enumerate(azimuth_ranges):
                self.motor_poses[:, i, 0:3] = position_from_spherical(
                    range=torch_rand_float(min_range, max_range, (self.num_samples,1), device=self.device),
                    azimuth=torch_rand_float(a0*torch.pi/180, a1*torch.pi/180, (self.num_samples,1), device=self.device),
                    elevation=torch_rand_float(min_pitch, max_pitch, (self.num_samples,1), device=self.device)
                ).squeeze(1)
            
            sampled_angles = torch.zeros(self.num_samples, self.motor_count, 3, device=self.device)
            sampled_angles[...,1] = 60 * torch.rand(self.num_samples, self.motor_count, device=self.device) # pitch tilt (30)
            sampled_angles[...,2] = torch.rand(self.num_samples, self.motor_count, device=self.device) * 360 #* 2 * torch.pi   # yaw
            sampled_angles = torch.deg2rad(sampled_angles)
            self.motor_poses[:, :, 3:7] = quat_from_euler_xyz_tensor(sampled_angles)




    def sample_airframes(self):
        """
        Sample airframes based on the configuration parameters.
        
        Returns:
            torch.Tensor: A tensor of sampled airframes with shape (num_samples, 6, motor_count).
        """

        if self.planar:
            if not self.wide:
                self.sampling_method_planar()
            else:
                self.sampling_method_planar_wide()
        elif self.symmetric:
            self.sampling_method_symmetric()
        elif self.random:
            self.sampling_method_random()
        else:
            self.sampling_method_all()

        # Sample robot mass and inertia
        self.robot_core_mass[:] = torch_rand_float_tensor(lower=self.robot_core_mass_min, upper=self.robot_core_mass_max)

        self.motor_mass[:] = torch_rand_float_tensor(lower=self.motor_mass_min, upper=self.motor_mass_max)


        # assume robot_core is at (0,0,0) in the robot frame. Calculate the robot COM position
        self.robot_com = torch.zeros(self.num_samples, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_core_pose[:, :3] = torch.zeros(self.num_samples, 3, dtype=torch.float32, requires_grad=False, device=self.device)
        self.robot_core_pose[:, 3:7] = 0.0
        self.robot_core_pose[:, 6] = 1.0  # Set the quaternion w component to 1.0 (no rotation)

        box_dims = torch_rand_float(lower=0.05, upper=0.15, shape=(self.num_samples, 3), device=self.device)
        length = box_dims[:, 0]
        width = box_dims[:, 1]
        height = box_dims[:, 2]

        self.robot_core_inertia = inertia_box_3d(
            mass=self.robot_core_mass,
            length=length,
            width=width,
            height=height
        )

        if self.payload:
            self.payload_mass = torch_rand_float(lower=0.2, upper=0.8, shape=(self.num_samples, 1), device=self.device)
        else:
            self.payload_mass = torch.zeros(self.num_samples, 1, dtype=torch.float32, device=self.device)
        box_dims = torch_rand_float(lower=0.05, upper=0.1, shape=(self.num_samples, 3), device=self.device)
        length = box_dims[:, 0]
        width = box_dims[:, 1]
        height = box_dims[:, 2]

        self.payload_poses = torch.zeros(self.num_samples, 7, dtype=torch.float32, requires_grad=False, device=self.device)
        self.payload_poses[:, 0:3] = position_from_spherical(
            range=torch_rand_float(lower=0.1, upper=0.2, shape=(self.num_samples, 1), device=self.device),
            azimuth=torch_rand_float(lower=0.0, upper=2*torch.pi, shape=(self.num_samples, 1), device=self.device),
            elevation=torch_rand_float(lower=-torch.pi/4, upper=torch.pi/4, shape=(self.num_samples, 1), device=self.device)
        ).squeeze(1)

        self.payload_poses[:, 3:7] = 0.0
        self.payload_poses[:, 6] = 1.0

        self.payload_inertia = inertia_box_3d(
            mass = self.payload_mass,
            length=length,
            width=width,
            height=height
        )

        self.robot_mass = self.robot_core_mass + self.motor_mass.sum(dim=1, keepdim=True) + self.payload_mass

        a1 = (self.motor_poses[:, :, :3] * self.motor_mass.unsqueeze(2)).sum(dim=1)
        a2 = self.robot_core_mass * self.robot_core_pose[:, :3]
        a3 = (self.payload_poses[:, :3] * self.payload_mass)

        self.robot_com = (a1 + a2 + a3) / self.robot_mass
        
        self.motor_pose_offsets = self.motor_poses[:, :, :3] - self.robot_com.unsqueeze(1)
        
        self.robot_inertia = self.calculate_robot_inertia()

        
        if self.random:
            self.robot_inertia[:] = torch. tensor([[ 1.5434e-03,  1.3366e-04,  5.7340e-05],
                [ 1.3366e-04,  1.4638e-03, -1.2388e-04],
                [ 5.7340e-05, -1.2388e-04,  2.1267e-03]],).unsqueeze(0).expand(self.num_samples, -1, -1)
        elif self.symmetric:
            self.robot_inertia[:] = torch.tensor([[0.7469,  0.0003,  0.0114],
                [0.0003,  0.7571, -0.0000],
                [0.0114, -0.0000,  0.9598]]).unsqueeze(0).expand(self.num_samples, -1, -1) * (10** -3)
        elif self.planar and not self.wide:
            self.robot_inertia[:] = torch.tensor([[ 1.6349e-03, -2.6919e-06,  7.0260e-07],
                [-2.6919e-06,  1.5729e-03, -1.4094e-05],
                [ 7.0260e-07, -1.4094e-05,  2.6225e-03]]).unsqueeze(0).expand(self.num_samples, -1, -1)
            


        self.motor_directions = -calculate_motor_directions(self.motor_poses_com[:, :, 0:3])
        self.allocation_matrix = calculate_allocation_matrix(
            self.motor_poses_com, self.motor_torque_constants, self.motor_directions)


    def return_valid_dict(self, valid_indices):
        valid_robot_inertias = self.robot_inertia[valid_indices].cpu().numpy()
        valid_robot_masses = self.robot_mass[valid_indices].cpu().numpy()
        valid_motor_inertias = self.motor_inertia[valid_indices].cpu().numpy()
        valid_motor_masses = self.motor_mass[valid_indices].cpu().numpy()
        valid_motor_force_constants = self.motor_force_constants[valid_indices].cpu().numpy()
        valid_motor_torque_constants = self.motor_torque_constants[valid_indices].cpu().numpy()
        valid_motor_directions = self.motor_directions[valid_indices].unsqueeze(2).cpu().numpy()
        valid_motor_poses = self.motor_poses[valid_indices].cpu().numpy()
        valid_motor_poses_com = self.motor_poses_com[valid_indices].cpu().numpy()
        valid_motor_min_thrusts = self.motor_min_thrusts[valid_indices].cpu().numpy()
        valid_motor_max_thrusts = self.motor_max_thrusts[valid_indices].cpu().numpy()
        valid_allocation_matrices = self.allocation_matrix[valid_indices].cpu().numpy()
        valid_allocation_matrices_pinv = torch.linalg.pinv(self.allocation_matrix[valid_indices]).cpu().numpy()
        valid_robot_inertias_inverse = torch.linalg.inv(self.robot_inertia[valid_indices]).cpu().numpy()
        valid_motor_time_constants = self.motor_time_constants[valid_indices].cpu().numpy()
        
        
        config_dict = {}
        config_dict["num_robots"] = len(valid_indices)
        config_dict["num_motors"] = self.motor_count
        config_dict["robot_mass"] = valid_robot_masses
        config_dict["robot_inertia"] = valid_robot_inertias
        config_dict["robot_inertia_inverse"] = valid_robot_inertias_inverse
        config_dict["allocation_matrix"] = valid_allocation_matrices
        config_dict["allocation_matrix_inverse"] = valid_allocation_matrices_pinv
        config_dict["motor_max_thrusts"] = valid_motor_max_thrusts
        config_dict["motor_min_thrusts"] = valid_motor_min_thrusts
        config_dict["motor_force_constants"] = valid_motor_force_constants
        config_dict["motor_torque_constants"] = valid_motor_torque_constants
        config_dict["motor_directions"] = valid_motor_directions
        config_dict["motor_poses"] = valid_motor_poses
        config_dict["motor_poses_com"] = valid_motor_poses_com
        config_dict["motor_inertia"] = valid_motor_inertias
        config_dict["motor_mass"] = valid_motor_masses
        config_dict['motor_time_constants'] = valid_motor_time_constants
        return config_dict
    
    def return_dict(self, valid_indices):
        full_indices = torch.arange(self.robot_mass.size(0), device=self.device)
        mask = torch.ones_like(full_indices, dtype=torch.bool)

        # 3. Mark the existing indices as False
        mask[valid_indices] = False

        # 4. Extract the complement indices
        valid_indices = full_indices[mask]
        valid_robot_inertias = self.robot_inertia[valid_indices].cpu().numpy()
        valid_robot_masses = self.robot_mass[valid_indices].cpu().numpy()
        valid_motor_inertias = self.motor_inertia[valid_indices].cpu().numpy()
        valid_motor_masses = self.motor_mass[valid_indices].cpu().numpy()
        valid_motor_force_constants = self.motor_force_constants[valid_indices].cpu().numpy()
        valid_motor_torque_constants = self.motor_torque_constants[valid_indices].cpu().numpy()
        valid_motor_directions = self.motor_directions[valid_indices].unsqueeze(2).cpu().numpy()
        valid_motor_poses = self.motor_poses[valid_indices].cpu().numpy()
        valid_motor_poses_com = self.motor_poses_com[valid_indices].cpu().numpy()
        valid_motor_min_thrusts = self.motor_min_thrusts[valid_indices].cpu().numpy()
        valid_motor_max_thrusts = self.motor_max_thrusts[valid_indices].cpu().numpy()
        valid_allocation_matrices = self.allocation_matrix[valid_indices].cpu().numpy()
        valid_allocation_matrices_pinv = torch.linalg.pinv(self.allocation_matrix[valid_indices]).cpu().numpy()
        valid_robot_inertias_inverse = torch.linalg.inv(self.robot_inertia[valid_indices]).cpu().numpy()
        valid_motor_time_constants = self.motor_time_constants[valid_indices].cpu().numpy()
        
        
        config_dict = {}
        config_dict["num_robots"] = len(valid_indices) 
        config_dict["num_motors"] = self.motor_count
        config_dict["robot_mass"] = valid_robot_masses
        config_dict["robot_inertia"] = valid_robot_inertias
        config_dict["robot_inertia_inverse"] = valid_robot_inertias_inverse
        config_dict["allocation_matrix"] = valid_allocation_matrices
        config_dict["allocation_matrix_inverse"] = valid_allocation_matrices_pinv
        config_dict["motor_max_thrusts"] = valid_motor_max_thrusts
        config_dict["motor_min_thrusts"] = valid_motor_min_thrusts
        config_dict["motor_force_constants"] = valid_motor_force_constants
        config_dict["motor_torque_constants"] = valid_motor_torque_constants
        config_dict["motor_directions"] = valid_motor_directions
        config_dict["motor_poses"] = valid_motor_poses
        config_dict["motor_poses_com"] = valid_motor_poses_com
        config_dict["motor_inertia"] = valid_motor_inertias
        config_dict["motor_mass"] = valid_motor_masses
        config_dict['motor_time_constants'] = valid_motor_time_constants
        return config_dict

    def load_config(self, config_dict):
        """
        Load the configuration from a dictionary.
        
        Args:
            config_dict (dict): A dictionary containing the configuration parameters.
        """
        self.num_samples = config_dict["num_robots"]
        self.motor_count = config_dict["num_motors"]

        self.robot_mass = torch.tensor(config_dict["robot_mass"], dtype=torch.float32, device=self.device)
        self.robot_inertia = torch.tensor(config_dict["robot_inertia"], dtype=torch.float32, device=self.device)
        self.robot_inertia_inv = torch.tensor(config_dict["robot_inertia_inverse"], dtype=torch.float32, device=self.device)
        self.allocation_matrix = torch.tensor(config_dict["allocation_matrix"], dtype=torch.float32, device=self.device)
        self.allocation_matrix_pinv = torch.tensor(config_dict["allocation_matrix_inverse"], dtype=torch.float32, device=self.device)
        
        self.motor_max_thrusts = torch.tensor(config_dict["motor_max_thrusts"], dtype=torch.float32, device=self.device)
        self.motor_min_thrusts = torch.tensor(config_dict["motor_min_thrusts"], dtype=torch.float32, device=self.device)
        self.motor_force_constants = torch.tensor(config_dict["motor_force_constants"], dtype=torch.float32, device=self.device)
        self.motor_torque_constants = torch.tensor(config_dict["motor_torque_constants"], dtype=torch.float32, device=self.device)
        
        self.motor_directions = torch.tensor(config_dict["motor_directions"], dtype=torch.float32, device=self.device)
        self.motor_poses = torch.tensor(config_dict["motor_poses"], dtype=torch.float32, device=self.device)
        
        self.motor_inertia = torch.tensor(config_dict["motor_inertia"], dtype=torch.float32, device=self.device)
        self.motor_mass = torch.tensor(config_dict["motor_mass"], dtype=torch.float32, device=self.device)

        self.motor_time_constants = torch.tensor(config_dict['motor_time_constants'], dtype=torch.float32, device=self.device)

    def validate_hover_feasibility(self,
        gravity_mag: float = 9.81,
        num_iterations: int = 3000,
        learning_rate: float = 0.05,
        force_penalty: float = 1000.0,
        torque_penalty: float = 10000.0,
        success_tolerance: float = 1e-5
        ):


        # --- Prepare inputs from consistent API ---
        FORCE_END_IDX = 3
        B1_batch = self.allocation_matrix[:, :FORCE_END_IDX, :]
        B2_batch = self.allocation_matrix[:, FORCE_END_IDX:, :]
        mg_batch = self.robot_mass.squeeze(-1) * gravity_mag

        # --- Initialize Optimization ---
        # Start controls in the middle of their valid range.
        u_batch = (self.motor_min_thrusts + self.motor_max_thrusts) / 2.0
        u_batch.requires_grad_(True)

        # Use Adam optimizer, which is generally robust for this type of problem.
        optimizer = Adam([u_batch], lr=learning_rate)

        # --- Optimization Loop (Projected Gradient Descent) ---
        for iter_num in range(num_iterations):
            optimizer.zero_grad()

            # 1. Objective: Minimize control effort (u^T * u)
            objective_loss = 0 * torch.sum(u_batch**2, dim=1)

            # 2. Force Constraint Penalty: ||B1*u|| should equal mg
            B1u = torch.bmm(B1_batch, u_batch.unsqueeze(2)).squeeze(2)
            force_constraint_loss = (torch.norm(B1u, dim=1) - mg_batch)**2
            

            # 3. Torque Constraint Penalty: B2*u should be zero
            B2u = torch.bmm(B2_batch, u_batch.unsqueeze(2)).squeeze(2)
            torque_constraint_loss = torch.sum(B2u**2, dim=1)
            
            # Total loss is a weighted sum of objective and penalties
            loss = (objective_loss + force_penalty * force_constraint_loss + torque_penalty * torque_constraint_loss).mean()
            
            loss.backward()
            optimizer.step()

            # Projection Step: Enforce box constraints by clamping
            with torch.no_grad():
                u_batch.data = torch.clamp(u_batch.data, self.motor_min_thrusts, self.motor_max_thrusts)

        optimal_thrusts = u_batch.detach()

        # --- Check for Success and Package Metrics ---
        final_force_vector = torch.bmm(B1_batch, optimal_thrusts.unsqueeze(2)).squeeze(2)
        final_torque_vector = torch.bmm(B2_batch, optimal_thrusts.unsqueeze(2)).squeeze(2)

        final_force_error = (torch.norm(final_force_vector, dim=1) - mg_batch)**2
        final_torque_error = torch.sum(final_torque_vector**2, dim=1)

        is_feasible = (final_force_error < success_tolerance) & (final_torque_error < success_tolerance)

        metrics = {
            "optimal_thrusts": optimal_thrusts,
            "final_force_error_sq": final_force_error,
            "final_torque_error_sq": final_torque_error,
            "final_force_vector": final_force_vector,
            "final_torque_vector": final_torque_vector,
        }
        
        return is_feasible, metrics
    

    @torch.no_grad()
    def validate_designs_tier1(
        self,
        min_twr: float = 2.0,
        min_max_angular_accel_rad_s2 = 80.0,
        max_condition_number: float = 10.0e22,
        gravity_mag: float = 9.81
        ):
        
        # --- 1. Control Authority: Condition Number ---
        # A low condition number implies the control problem is well-posed and stable.
        condition_numbers = torch.linalg.cond(self.allocation_matrix)
        is_valid_by_cond = condition_numbers < max_condition_number

        # --- 2. Vertical Thrust: Thrust-to-Weight Ratio (TWR) ---
        # To maximize vertical thrust, use max/min motor limits based on their contribution.
        FORCE_Z_IDX = 2
        z_force_coeffs = self.allocation_matrix[:, FORCE_Z_IDX, :]
        
        u_for_max_thrust = torch.where(z_force_coeffs > 0, self.motor_max_thrusts, self.motor_min_thrusts)
        max_vertical_thrust = torch.einsum('bi,bi->b', z_force_coeffs, u_for_max_thrust)
        
        robot_weight = self.robot_mass.squeeze(1) * gravity_mag
        twr = max_vertical_thrust / robot_weight
        is_valid_by_twr = twr > min_twr
        print(z_force_coeffs.shape, robot_weight.shape,u_for_max_thrust.shape )
        print(twr.shape, is_valid_by_twr.shape, twr[0], is_valid_by_twr[0])
        # --- 3. Angular Maneuverability: Max Angular Acceleration (Vectorized) ---
        # Calculate max possible torque magnitude along each axis (roll, pitch, yaw) simultaneously.
        TORQUE_START_IDX = 3
        torque_coeffs = self.allocation_matrix[:, TORQUE_START_IDX:, :]  # Shape: (N, 3, M)

        # Unsqueeze motor limits to enable broadcasting with torque_coeffs
        motor_max_unsqueezed = self.motor_max_thrusts.unsqueeze(1) # Shape: (N, 1, M)
        motor_min_unsqueezed = self.motor_min_thrusts.unsqueeze(1) # Shape: (N, 1, M)

        # Determine motor commands for max positive and negative torque on each axis
        u_for_max_pos_torque = torch.where(torque_coeffs > 0, motor_max_unsqueezed, motor_min_unsqueezed)
        u_for_max_neg_torque = torch.where(torque_coeffs < 0, motor_max_unsqueezed, motor_min_unsqueezed)

        # Calculate max torques using batched dot product (einsum)
        max_pos_torques = torch.einsum('bij,bij->bi', torque_coeffs, u_for_max_pos_torque) # Shape: (N, 3)
        max_neg_torques = torch.einsum('bij,bij->bi', torque_coeffs, u_for_max_neg_torque) # Shape: (N, 3)

        # Maneuverability is limited by the weaker direction (e.g., can you roll right as fast as left?)
        # We care about the magnitude, so we use min(positive_torque, |negative_torque|).
        max_abs_torques = torch.min(max_pos_torques, -max_neg_torques)

        # Convert max torque to max angular acceleration (alpha = I_inv @ tau)
        try:
            robot_inertia_inv = torch.linalg.inv(self.robot_inertia)
            max_angular_accels = torch.bmm(robot_inertia_inv, max_abs_torques.unsqueeze(-1)).squeeze(-1)
        except torch.linalg.LinAlgError:
            # print("Warning: Singular inertia matrix detected. Setting accelerations to zero for these designs.")
            max_angular_accels = torch.zeros_like(max_abs_torques)

        # Check if all axes meet the minimum acceleration requirement
        if isinstance(min_max_angular_accel_rad_s2, (list, tuple)):
            min_accel_tensor = torch.tensor(min_max_angular_accel_rad_s2, device=self.device).expand_as(max_angular_accels)
        else:
            min_accel_tensor = min_max_angular_accel_rad_s2
        print('ang accel', max_angular_accels[0], min_accel_tensor,  max_angular_accels.shape)
            
        is_valid_by_accel = torch.all(max_angular_accels > min_accel_tensor, dim=1)
        print(max_angular_accels[0], is_valid_by_accel[0], max_angular_accels.shape)
        # --- Final Validation & Metrics ---
        is_fully_valid = is_valid_by_cond & is_valid_by_twr & is_valid_by_accel

        metrics = {
            "condition_number": condition_numbers,
            "twr": twr,
            "max_angular_acceleration_xyz": max_angular_accels,
            "is_valid_by_condition_number": is_valid_by_cond,
            "is_valid_by_twr": is_valid_by_twr,
            "is_valid_by_angular_accel": is_valid_by_accel,
        }

        return is_fully_valid, metrics
    

    

    
if __name__ == "__main__":
    planar = False
    random = False
    symmetric = False
    wide = False
    motor_count = 6
    if planar:
        save_name = "planar"
    elif symmetric:
        save_name = "symmetric"
    elif random:
        save_name = "random"
    else:
        save_name = "all"
    if wide and not (save_name == "all"):
        save_name += "_wide"
    save_name = "valid_airframe_config_" + str(motor_count) + "_" + save_name + ".pkl"
    sampler = AirframeSamplerCfg(
        num_samples=2*4096 if planar or random or symmetric else 16384,
        random=random,
        planar=planar,
        symmetric=symmetric,
        wide=wide,
        payload=False,
        motor_count=motor_count
    )

    # print(sampler.motor_min_thrusts.mean().item(), sampler.motor_max_thrusts.mean().item())
    
    start_time = time.time()
    sampler.sample_airframes()
    sampler.motor_min_thrusts[:] = 0.0
    sampler.motor_max_thrusts[:] = sampler.robot_mass*9.81/2 #8.0
    results = sampler.validate_hover_feasibility(
        gravity_mag=9.81,
        num_iterations=30000,
        learning_rate=0.005,
        force_penalty=1000.0,
        torque_penalty=10000.0,
        success_tolerance=1e-5
    )

    end_time = time.time()


    valid_designs_check1 = results[0]

    percentage_feasible = results[0].float().mean().item() * 100
    print(f"Percentage of feasible designs: {percentage_feasible:.2f}%")
    print(f"Time taken for validation: {end_time - start_time:.2f} seconds")


    start_time = time.time()


    results2 = sampler.validate_designs_tier1(
        min_twr=2.5, #2.5
        min_max_angular_accel_rad_s2=16.0,
        max_condition_number=10.0e22,
        gravity_mag=9.81
    )

    end_time = time.time()

    valid_designs_check2 = results2[0]


    percentage_valid = results2[0].float().mean().item() * 100
    print(f"Percentage of valid designs: {percentage_valid:.2f}%")
    print("Time taken for Tier 1 validation:", end_time - start_time, "seconds")
    print("is valid by condition number", results2[1]["is_valid_by_condition_number"].float().mean().item() * 100)
    print("is valid by TWR", results2[1]["is_valid_by_twr"].float().mean().item() * 100)
    print("is valid by angular accel", results2[1]["is_valid_by_angular_accel"].float().mean().item() * 100)
   

    # check both results and select ones where both are feasible
    valid_designs = results[0] & results2[0]
    print('0 is valid', valid_designs[0].item())
    percentage_valid_designs = valid_designs.float().mean().item() * 100
    print(f"Percentage of designs valid in both Tier 1 and feasibility checks: {percentage_valid_designs:.2f}%")


    # print the indices of robot designs that are valid in both checks
    valid_indices = torch.nonzero(valid_designs).squeeze()
    # print("Indices of valid designs in both checks:", valid_indices.tolist())

    valid_config_dict = sampler.return_valid_dict(valid_indices)
    config_dict = sampler.return_dict(valid_indices)


    print(f"Number of valid designs: {len(valid_indices)}")

    # save config_dict to a file
    import pickle as pkl

    with open(save_name, "wb") as f:
        pkl.dump(valid_config_dict, f)
    # with open("airframe_config_6_new_m_big.pkl", "wb") as f:
    #     pkl.dump(config_dict, f)
    print(valid_config_dict["allocation_matrix"][0])
    print(f"Valid airframe configuration saved to {save_name}")

    # stastics of the robots that failed and the reasons for Tier 1 validation
    is_valid_condition_numbers = results2[1]["is_valid_by_condition_number"]
    is_valid_twr = results2[1]["is_valid_by_twr"]
    is_valid_max_angular_accel = results2[1]["is_valid_by_angular_accel"]
    print(f"Number of designs that failed Tier 1 validation: {len(valid_indices) - valid_designs.sum().item()}")
    failed_indices = torch.nonzero(~valid_designs).squeeze()
    print(f"Number of designs that failed condition number check: {len(torch.nonzero(~is_valid_condition_numbers).squeeze())}")
    print(f"Number of designs that failed thrust-to-weight ratio check: {len(torch.nonzero(~is_valid_twr).squeeze())}")
    print(f"Number of designs that failed angular acceleration check: {len(torch.nonzero(~is_valid_max_angular_accel).squeeze())}")

    print(sampler.robot_mass[0], sampler.robot_inertia[0], sampler.robot_core_inertia[0])




