import numpy as np
import cvxpy as cp

# =============================
# Geometry utilities
# =============================

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])

def transform_vertices(V, p, q):
    R = quat_to_rot(q)
    return (R @ V.T).T + p

# =============================
# DCOL (linearized, external)
# =============================

def dcol_linearize(Vc, Vt, pc, qc, pt, qt):
    """
    External proximity computation (GJK-style surrogate).
    Returns:
        alpha0 : signed distance
        n      : separating normal (w.r.t chaser position)
    """

    Vc_w = transform_vertices(Vc, pc, qc)
    Vt_w = transform_vertices(Vt, pt, qt)

    # Brute-force closest pair (OK for small hulls)
    dmin = 1e9
    pa, pb = None, None
    for a in Vc_w:
        for b in Vt_w:
            d = np.linalg.norm(a - b)
            if d < dmin:
                dmin = d
                pa, pb = a, b

    n = (pa - pb)
    n = n / (np.linalg.norm(n) + 1e-8)

    return dmin, n


# =============================
# SCP Docking Planner (13D)
# =============================

def plan_scp_13d_dcol(
    x0,
    p_target,
    q_target,
    V_chaser,
    V_target,
    N=80,
    dt=0.1,
    alpha_min=0.06,
    trust_radius=0.15,
    scp_iters=6,
    psi=1000.0
):
    """
    x = [p(3), v(3), q(4), w(3)]
    """

    nx = 13
    nu = 6

    # Initial straight-line guess
    x_ref = np.tile(x0.reshape(-1,1), (1,N))
    for k in range(N):
        a = k/(N-1)
        x_ref[0:3,k] = (1-a)*x0[0:3] + a*p_target
        x_ref[6:10,k] = q_target

    for _ in range(scp_iters):

        x = cp.Variable((nx, N))
        u = cp.Variable((nu, N-1))
        s = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + psi * cp.sum(s)
        cons = [x[:,0] == x0]

        # Dynamics
        for k in range(N-1):
            cons += [
                x[0:3,k+1] == x[0:3,k] + dt * x[3:6,k],
                x[3:6,k+1] == x[3:6,k] + dt * u[0:3,k],
                x[6:10,k+1] == x[6:10,k] + dt * cp.hstack([0, u[3:6,k]]),
                x[10:13,k+1] == u[3:6,k]
            ]

        # Terminal docking
        cons += [
            cp.norm(x[0:3,-1] - p_target) <= 0.03,
            cp.norm(x[6:10,-1] - q_target) <= 0.05
        ]

        # DCOL constraints (linearized)
        for k in range(N):
            pc = x_ref[0:3,k]
            qc = x_ref[6:10,k]

            alpha0, n = dcol_linearize(
                V_chaser, V_target,
                pc, qc,
                p_target, q_target
            )

            cons += [
                alpha0 + n @ (x[0:3,k] - pc) + s[k] >= alpha_min,
                cp.norm(x[:,k] - x_ref[:,k]) <= trust_radius
            ]

        cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.ECOS)

        if x.value is None:
            break

        x_ref = x.value.copy()

    return x_ref
