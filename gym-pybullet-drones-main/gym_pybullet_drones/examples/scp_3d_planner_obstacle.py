import numpy as np
import cvxpy as cp

def plan_scp_trajectory(
    p_start,
    p_goal,
    p_obs,
    r_obs,
    N=80,
    dt=0.1
):
    # ----------------------------
    # Dynamics (3D double integrator)
    # ----------------------------
    A = np.eye(6)
    A[0,3] = dt; A[1,4] = dt; A[2,5] = dt

    B = np.zeros((6,3))
    B[0,0] = 0.5*dt**2
    B[1,1] = 0.5*dt**2
    B[2,2] = 0.5*dt**2
    B[3,0] = dt
    B[4,1] = dt
    B[5,2] = dt

    # ----------------------------
    # Initial guess (curved to break symmetry)
    # ----------------------------
    x_ref = np.zeros((6, N))
    for k in range(N):
        alpha = k/(N-1)
        base = (1-alpha)*p_start + alpha*p_goal
        bump = np.array([0.0, 1.5*np.sin(np.pi*alpha), 0.8*np.sin(np.pi*alpha)])
        x_ref[0:3,k] = base + bump

    max_iters = 12
    for _ in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + 1e5 * cp.sum(slack)
        constraints = []

        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3)])]

        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

        constraints += [
            cp.norm(x[0:3,-1] - p_goal) <= 0.05,
            x[3:6,-1] == 0
        ]

        # ----------------------------
        # Obstacle linearization
        # ----------------------------
        for k in range(1, N-1):
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)

            if dist < 1e-3:
                n = np.array([0., 1., 0.])
            else:
                n = vec / dist

            constraints += [
                n @ (x[0:3,k] - p_obs) >= r_obs - slack[k]
            ]

        # Trust region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.8]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)

        if x.value is None:
            raise RuntimeError("SCP failed")

        x_ref = x.value.copy()

    return x_ref[0:3,:].T
