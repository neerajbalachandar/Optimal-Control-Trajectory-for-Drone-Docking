import numpy as np
import cvxpy as cp
from collision_models import signed_distance_oriented_box

def plan_scp_13d_docking(
    x0,
    p_target,
    q_target,
    box_chaser,
    box_target,
    N=200,
    dt=0.1,
    d_min=0.05,
    trust_radius=0.1,
    psi=1000.0,
    scp_iters=6
):
    """
    x = [p(3), v(3), q(4), w(3)]
    """

    # Reference
    x_ref = np.tile(x0.reshape(-1,1), (1,N))
    for k in range(N):
        a = k/(N-1)
        x_ref[0:3,k] = (1-a)*x0[0:3] + a*p_target

    for _ in range(scp_iters):
        x = cp.Variable((13,N))
        u = cp.Variable((6,N-1))  # accel + ang vel
        slack = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + psi * cp.sum(slack)
        cons = [x[:,0] == x0]

        for k in range(N-1):
            # translation
            cons += [x[0:3,k+1] == x[0:3,k] + dt*x[3:6,k]]
            cons += [x[3:6,k+1] == x[3:6,k] + dt*u[0:3,k]]

            # quaternion kinematics
            cons += [x[6:10,k+1] == x[6:10,k] + dt*cp.hstack([0, u[3:6,k]])]

            # angular velocity
            cons += [x[10:13,k+1] == u[3:6,k]]

        # terminal docking
        cons += [
            cp.norm(x[0:3,-1] - p_target) <= 0.03,
            cp.norm(x[6:10,-1] - q_target) <= 0.05
        ]

        # DCOL linearized
        for k in range(N):
            pref = x_ref[0:3,k]
            qref = x_ref[6:10,k]

            d0 = signed_distance_oriented_box(
                pref, p_target, qref, box_target
            )

            n = (pref - p_target)
            n = n / (np.linalg.norm(n)+1e-6)

            cons += [
                d0 + n @ (x[0:3,k] - pref) + slack[k] >= d_min,
                cp.norm(x[:,k] - x_ref[:,k]) <= trust_radius
            ]

        cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.ECOS)

        if x.value is None:
            break
        x_ref = x.value.copy()

    return x_ref
