import numpy as np
import cvxpy as cp

def omega_matrix(w):
    wx, wy, wz = w
    return np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0 ]
    ])

def plan_scp_13d(
    p_start,
    q_start,
    p_goal,
    q_goal,
    p_obs,
    r_obs,
    N=60,
    dt=0.1
):
    # State: [p(3), v(3), q(4), w(3)]
    nx = 13
    nu = 6  # [a(3), tau(3)]

    # Initial reference
    x_ref = np.zeros((nx, N))
    for k in range(N):
        alpha = k/(N-1)
        x_ref[0:3,k] = (1-alpha)*p_start + alpha*p_goal
        x_ref[6:10,k] = (1-alpha)*q_start + alpha*q_goal

    max_iters = 10

    for _ in range(max_iters):
        x = cp.Variable((nx, N))
        u = cp.Variable((nu, N-1))
        slack = cp.Variable(N, nonneg=True)

        cost = (
            cp.sum_squares(u[0:3, :]) +
            cp.sum_squares(u[3:6, :]) +
            1e5 * cp.sum(slack)
        )

        constraints = []

        # Initial condition
        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3), q_start, np.zeros(3)])]

        for k in range(N-1):
            # Translational dynamics
            constraints += [
                x[0:3,k+1] == x[0:3,k] + x[3:6,k]*dt + 0.5*u[0:3,k]*dt**2,
                x[3:6,k+1] == x[3:6,k] + u[0:3,k]*dt
            ]

            # Angular velocity
            constraints += [
                x[10:13,k+1] == x[10:13,k] + u[3:6,k]*dt
            ]

            # Quaternion (linearized)
            Omega = omega_matrix(x_ref[10:13,k])
            constraints += [
                x[6:10,k+1] == x[6:10,k] + 0.5 * Omega @ x[6:10,k] * dt
            ]

        # Terminal docking constraints
        constraints += [
            cp.norm(x[0:3,-1] - p_goal) <= 0.05,
            x[3:6,-1] == 0,
            x[10:13,-1] == 0
        ]

        # Obstacle avoidance
        for k in range(1, N-1):
            pref = x_ref[0:3,k]
            vec = pref - p_obs
            dist = np.linalg.norm(vec)
            n = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [
                n @ (x[0:3,k] - p_obs) >= r_obs - slack[k]
            ]

        # Trust region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.7]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)

        if x.value is None:
            raise RuntimeError("SCP failed")

        x_ref = x.value.copy()

    return x_ref
