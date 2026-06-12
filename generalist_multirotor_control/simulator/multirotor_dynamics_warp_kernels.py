import warp as wp
from generalist_multirotor_control.utils.warp_math import *

wp.init()

NUM_MOTORS = 6 # Updated to 4 motors

@wp.func
def quadrotor_dynamics_derivative(
    pos: wp.vec3f,
    quat: wp.vec4f,
    vel: wp.vec3f,
    angvel: wp.vec3f,
    motor_rpm: wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32),
    motor_setpoint: wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32),
    motor_dir: wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32),
    motor_force_constant: wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32),
    motor_time_constant: wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32),
    mass: wp.float32,
    inertia: wp.mat(shape=(3, 3), dtype=wp.float32),
    inertia_inv: wp.mat(shape=(3, 3), dtype=wp.float32),
    gravity: wp.vec3f,
    allocation_matrix: wp.mat(shape=(6, NUM_MOTORS), dtype=wp.float32),
):
    d_pos = vel

    angvel_body = quat_rotate_inverse(quat, angvel)

    # Compute motor wrench (forces and torques)
    applied_wrench = allocation_matrix @ wp.cw_mul(
        motor_force_constant, wp.cw_mul(motor_rpm, motor_rpm)
    )  # allocation_matmul(allocation_matrix, motor_rpm)

    # Velocity derivative
    force_body = wp.vec3(applied_wrench[0, 0], applied_wrench[1, 0], applied_wrench[2, 0])
    d_vel = (1.0 / mass) * (quat_rotate(quat, force_body)) + gravity

    # Quaternion derivative
    d_quat = quat_derivative(quat, angvel_body)

    # Angular velocity derivative
    torque_body = wp.vec3(applied_wrench[3, 0], applied_wrench[4, 0], applied_wrench[5, 0])
    inertia_times_omega = wp.mul(inertia, angvel_body)
    gyro_torque = wp.cross(angvel_body, inertia_times_omega)
    net_torque = torque_body - gyro_torque

    d_angvel_body = wp.mul(inertia_inv, net_torque)
    d_angvel = quat_rotate(quat, d_angvel_body)
    # Motor RPM derivative
    d_motor_rpm = wp.cw_div((motor_setpoint - motor_rpm), motor_time_constant)
    return d_pos, d_quat, d_vel, d_angvel, d_motor_rpm


@wp.kernel
def euler_quad_dynamics_update(
    # allocation_matrix: wp.mat(shape=(6,4), dtype=wp.float32),
    pos: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.vec4f),
    vel: wp.array(dtype=wp.vec3f),
    angvel: wp.array(dtype=wp.vec3f),
    motor_rpm: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_setpoint: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_dir: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_force_constant: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_time_constant: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat(shape=(3, 3), dtype=wp.float32)),
    inertia_inv: wp.array(dtype=wp.mat(shape=(3, 3), dtype=wp.float32)),
    gravity: wp.array(dtype=wp.vec3f),
    allocation_matrix: wp.array(dtype=wp.mat(shape=(6, NUM_MOTORS), dtype=wp.float32)),
    dt: float,
):
    tid = wp.tid()

    # Compute derivatives
    d_pos, d_quat, d_vel, d_angvel, d_motor_rpm = quadrotor_dynamics_derivative(
        pos[tid],
        quat[tid],
        vel[tid],
        angvel[tid],
        motor_rpm[tid],
        motor_setpoint[tid],
        motor_dir[tid],
        motor_force_constant[tid],
        motor_time_constant[tid],
        mass[tid],
        inertia[tid],
        inertia_inv[tid],
        gravity[tid],
        allocation_matrix[tid],
    )

    # Euler integration
    pos[tid] = pos[tid] + dt * d_pos
    quat[tid] = wp.normalize(quat[tid] + dt * d_quat)
    vel[tid] = vel[tid] + dt * d_vel
    angvel[tid] = angvel[tid] + dt * d_angvel
    motor_rpm[tid] = motor_rpm[tid] + dt * d_motor_rpm


@wp.kernel
def rk4_quad_dynamics_update(
    # allocation_matrix: wp.mat(shape=(6,4), dtype=wp.float32),
    pos: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.vec4f),
    vel: wp.array(dtype=wp.vec3f),
    angvel: wp.array(dtype=wp.vec3f),
    motor_rpm: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_setpoint: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_dir: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_force_constant: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    motor_time_constant: wp.array(dtype=wp.mat(shape=(NUM_MOTORS, 1), dtype=wp.float32)),
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat(shape=(3, 3), dtype=wp.float32)),
    inertia_inv: wp.array(dtype=wp.mat(shape=(3, 3), dtype=wp.float32)),
    gravity: wp.array(dtype=wp.vec3f),
    allocation_matrix: wp.array(dtype=wp.mat(shape=(6, NUM_MOTORS), dtype=wp.float32)),
    dt: float,
):

    tid = wp.tid()

    # Compute derivatives
    d_pos1, d_quat1, d_vel1, d_angvel1, d_motor_rpm1 = quadrotor_dynamics_derivative(
        pos[tid],
        quat[tid],
        vel[tid],
        angvel[tid],
        motor_rpm[tid],
        motor_setpoint[tid],
        motor_dir[tid],
        motor_force_constant[tid],
        motor_time_constant[tid],
        mass[tid],
        inertia[tid],
        inertia_inv[tid],
        gravity[tid],
        allocation_matrix[tid],
    )

    pos1 = pos[tid] + 0.5 * dt * d_pos1
    quat1 = wp.normalize(quat[tid] + 0.5 * dt * d_quat1)
    vel1 = vel[tid] + 0.5 * dt * d_vel1
    angvel1 = angvel[tid] + 0.5 * dt * d_angvel1
    motor_rpm1 = motor_rpm[tid] + 0.5 * dt * d_motor_rpm1

    d_pos2, d_quat2, d_vel2, d_angvel2, d_motor_rpm2 = quadrotor_dynamics_derivative(
        pos1,
        quat1,
        vel1,
        angvel1,
        motor_rpm1,
        motor_setpoint[tid],
        motor_dir[tid],
        motor_force_constant[tid],
        motor_time_constant[tid],
        mass[tid],
        inertia[tid],
        inertia_inv[tid],
        gravity[tid],
        allocation_matrix[tid],
    )

    pos2 = pos[tid] + 0.5 * dt * d_pos2
    quat2 = wp.normalize(quat[tid] + 0.5 * dt * d_quat2)
    vel2 = vel[tid] + 0.5 * dt * d_vel2
    angvel2 = angvel[tid] + 0.5 * dt * d_angvel2
    motor_rpm2 = motor_rpm[tid] + 0.5 * dt * d_motor_rpm2

    d_pos3, d_quat3, d_vel3, d_angvel3, d_motor_rpm3 = quadrotor_dynamics_derivative(
        pos2,
        quat2,
        vel2,
        angvel2,
        motor_rpm2,
        motor_setpoint[tid],
        motor_dir[tid],
        motor_force_constant[tid],
        motor_time_constant[tid],
        mass[tid],
        inertia[tid],
        inertia_inv[tid],
        gravity[tid],
        allocation_matrix[tid],
    )

    pos3 = pos[tid] + dt * d_pos3
    quat3 = wp.normalize(quat[tid] + dt * d_quat3)
    vel3 = vel[tid] + dt * d_vel3
    angvel3 = angvel[tid] + dt * d_angvel3
    motor_rpm3 = motor_rpm[tid] + dt * d_motor_rpm3

    d_pos4, d_quat4, d_vel4, d_angvel4, d_motor_rpm4 = quadrotor_dynamics_derivative(
        pos3,
        quat3,
        vel3,
        angvel3,
        motor_rpm3,
        motor_setpoint[tid],
        motor_dir[tid],
        motor_force_constant[tid],
        motor_time_constant[tid],
        mass[tid],
        inertia[tid],
        inertia_inv[tid],
        gravity[tid],
        allocation_matrix[tid],
    )

    pos[tid] = pos[tid] + (dt / 6.0) * (d_pos1 + 2.0 * d_pos2 + 2.0 * d_pos3 + d_pos4)
    quat[tid] = wp.normalize(
        quat[tid] + (dt / 6.0) * (d_quat1 + 2.0 * d_quat2 + 2.0 * d_quat3 + d_quat4)
    )
    vel[tid] = vel[tid] + (dt / 6.0) * (d_vel1 + 2.0 * d_vel2 + 2.0 * d_vel3 + d_vel4)
    angvel[tid] = angvel[tid] + (dt / 6.0) * (
        d_angvel1 + 2.0 * d_angvel2 + 2.0 * d_angvel3 + d_angvel4
    )
    motor_rpm[tid] = motor_rpm[tid] + (dt / 6.0) * (
        d_motor_rpm1 + 2.0 * d_motor_rpm2 + 2.0 * d_motor_rpm3 + d_motor_rpm4
    )
