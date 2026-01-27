import numpy as np
import cvxpy as cp

# ===============================
# DCOL metric (sphere hulls)
# ===============================
def alpha_dcol(p_c, p_t, r_c, r_t):
    return np.linalg.norm(p_c - p_t) / (r_c + r_t)

# ===============================
# Problem 2: Initial docking SCP
# ===============================
def solve_initial_docking(p0, p_goal, axis, cone_angle, N, dt):
    A = np.eye(6)
    A[0:3,3:6] = dt*np.eye(3)
    B = np.zeros((6,3))
    B[0:3,:] = 0.5*dt**2*np.eye(3)
    B[3:6,:] = dt*np.eye(3)

    n = axis / np.linalg.norm(axis)
    cos_th = np.cos(np.deg2rad(cone_angle))

    x = cp.Variable((6,N))
    u = cp.Variable((3,N-1))

    cost = cp.sum_squares(u)
    cons = [x[:,0] == np.hstack([p0, np.zeros(3)])]

    for k in range(N-1):
        cons += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

    # docking cone
    for k in range(N):
        p_rel = x[0:3,k] - p_goal
        dist_long = -n @ p_rel
        cons += [
            dist_long >= 0,
            cp.norm(p_rel)*cos_th <= dist_long
        ]

    cons += [
        cp.norm(x[0:3,-1] - p_goal) <= 0.05,
        cp.norm(x[3:6,-1]) <= 0.05
    ]

    cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.ECOS)
    return x.value

# ===============================
# Problem 3: SCP + DCOL
# ===============================
def solve_scp_dcol(x_ref, p_target, r_c, r_t, N, dt, psi=1000, iters=6):
    for _ in range(iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        s = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + psi*cp.sum(s)
        cons = [x[:,0] == x_ref[:,0]]

        for k in range(N-1):
            cons += [x[:,k+1] == x_ref[:,k+1] + (x[:,k]-x_ref[:,k])]

        for k in range(N):
            p_ref = x_ref[0:3,k]
            d = np.linalg.norm(p_ref - p_target)
            n = (p_ref - p_target)/(d+1e-6)
            alpha_bar = d/(r_c+r_t)
            J = n/(r_c+r_t)

            cons += [
                alpha_bar + J @ (x[0:3,k]-p_ref) + s[k] >= 1.0,
                cp.norm(x[:,k]-x_ref[:,k]) <= 0.5
            ]

        cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.ECOS)
        if x.value is None:
            break
        x_ref = x.value.copy()

    return x_ref
