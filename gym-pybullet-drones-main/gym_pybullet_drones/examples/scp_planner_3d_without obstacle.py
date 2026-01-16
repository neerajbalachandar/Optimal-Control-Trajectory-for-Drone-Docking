import numpy as np
import cvxpy as cp

def plan_scp_trajectory(
    p_start,
    p_goal,
    N=80,
    dt=0.1
):
    # 3D double integrator
    A = np.eye(6)
    A[0,3] = dt; A[1,4] = dt; A[2,5] = dt

    B = np.zeros((6,3))
    B[0,0] = 0.5*dt**2
    B[1,1] = 0.5*dt**2
    B[2,2] = 0.5*dt**2
    B[3,0] = dt
    B[4,1] = dt
    B[5,2] = dt

    # Initial guess: straight line
    x_ref = np.zeros((6, N))
    for k in range(N):
        alpha = k/(N-1)
        x_ref[0:3,k] = (1-alpha)*p_start + alpha*p_goal

    max_iters = 10
    for _ in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))

        cost = cp.sum_squares(u)
        constraints = []

        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3)])]

        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

        constraints += [
            cp.norm(x[0:3,-1] - p_goal) <= 0.05,
            x[3:6,-1] == 0
        ]

        # Trust region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS)

        if x.value is None:
            raise RuntimeError("SCP failed")

        x_ref = x.value.copy()

    # Return position waypoints only
    return x_ref[0:3,:].T
