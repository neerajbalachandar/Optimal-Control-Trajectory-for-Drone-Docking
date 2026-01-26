import numpy as np
import cvxpy as cp

# -------------------------------
# Quaternion utilities
# -------------------------------
def quat_from_two_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if c < -0.999999:
        return np.array([0, 1, 0, 0])

    s = np.sqrt((1 + c) * 2)
    q = np.array([s * 0.5, v[0] / s, v[1] / s, v[2] / s])
    return q / np.linalg.norm(q)


def compute_attitude_trajectory(p_traj, p_target):
    q_traj = np.zeros((4, p_traj.shape[1]))
    ez = np.array([0, 0, 1])

    for k in range(p_traj.shape[1]):
        d = p_target - p_traj[:, k]
        if np.linalg.norm(d) < 1e-6:
            q_traj[:, k] = np.array([1, 0, 0, 0])
        else:
            q_traj[:, k] = quat_from_two_vectors(ez, d)

    return q_traj


# -------------------------------
# SCP + DCOL Planner (POSE-AWARE)
# -------------------------------
def plan_scp_3d_pose_dcol(
    p_start,
    p_target,
    r_c=0.25,
    r_t=0.25,
    N=60,
    dt=0.1,
    psi=800.0,
    trust_radius=0.6,
    scp_iters=6
):
    # 6D translational dynamics
    A = np.eye(6)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt

    B = np.zeros((6, 3))
    B[0:3, :] = 0.5 * dt**2 * np.eye(3)
    B[3:6, :] = dt * np.eye(3)

    # Initial straight-line reference
    x_ref = np.zeros((6, N))
    for k in range(N):
        a = k / (N - 1)
        x_ref[0:3, k] = (1 - a) * p_start + a * p_target

    # SCP loop
    for _ in range(scp_iters):
        x = cp.Variable((6, N))
        u = cp.Variable((3, N - 1))
        slack = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + psi * cp.sum(slack)
        constraints = []

        constraints += [x[:, 0] == np.hstack([p_start, np.zeros(3)])]

        for k in range(N - 1):
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]

        constraints += [
            cp.norm(x[0:3, -1] - p_target) <= 0.05,
            x[3:6, -1] == 0
        ]

        # Linearized DCOL
        for k in range(N):
            pref = x_ref[0:3, k]
            d = np.linalg.norm(pref - p_target)
            n = (pref - p_target) / (d + 1e-6)

            alpha_bar = d / (r_c + r_t)
            J = n / (r_c + r_t)

            constraints += [
                alpha_bar + J @ (x[0:3, k] - pref) + slack[k] >= 1.0,
                cp.norm(x[:, k] - x_ref[:, k]) <= trust_radius
            ]

        cp.Problem(cp.Minimize(cost), constraints).solve(solver=cp.ECOS)

        if x.value is None:
            break

        x_ref = x.value.copy()

    # Generate attitude AFTER SCP
    q_traj = compute_attitude_trajectory(x_ref[0:3, :], p_target)

    return x_ref, q_traj
