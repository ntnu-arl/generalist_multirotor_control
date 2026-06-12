import warp as wp



@wp.func
def quat_mul(q1: wp.vec4f, q2: wp.vec4f):
    q1x = q1[0]
    q1y = q1[1]
    q1z = q1[2]
    q1w = q1[3]

    q2x = q2[0]
    q2y = q2[1]
    q2z = q2[2]
    q2w = q2[3]

    return wp.vec4f(
        q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
        q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
        q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
        q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
    )


@wp.func
def quat_rotate(quat: wp.vec4f, vec: wp.vec3f):
    q2 = wp.quatf(quat[0], quat[1], quat[2], quat[3])
    return wp.quat_rotate(q2, vec)


@wp.func
def quat_rotate_inverse(quat: wp.vec4f, vec: wp.vec3f):
    q2 = wp.quatf(quat[0], quat[1], quat[2], quat[3])
    return wp.quat_rotate_inv(q2, vec)


@wp.func
def quat_derivative(quat: wp.vec4f, angvel_body: wp.vec3f):
    quat_omega = 0.5 * wp.vec4f(angvel_body[0], angvel_body[1], angvel_body[2], 0.0)

    q_omega = wp.quatf(quat_omega[0], quat_omega[1], quat_omega[2], quat_omega[3])
    q2 = wp.quatf(quat[0], quat[1], quat[2], quat[3])
    q_res = q2 * q_omega
    return wp.vec4f(q_res[0], q_res[1], q_res[2], q_res[3])
