import time
import numpy as np
import cvxpy as cp
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ======================================================================
# CONFIGURATION
# ======================================================================
DOCKING_AXIS = np.array([0.0, 0.0, -1.0])
CONE_ANGLE   = 30.0
CTRL_FREQ    = 48
SIM_FREQ     = 240
DT           = 1.0 / CTRL_FREQ
HORIZON_SEC  = 20
N            = CTRL_FREQ * HORIZON_SEC

SAFETY_R     = 0.1
ALPHA_LIMIT  = 0.9

# ======================================================================
# DCOL CHECK
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    alpha = dist / (r1 + r2)
    contact = p1 + (p2 - p1) * (r1 / (r1 + r2))
    return alpha, contact

# ======================================================================
# SCP PLANNER
# ======================================================================
def plan_scp_docking(
    p_start, v_start, p_goal, p_obs, r_obs,
    docking_axis, cone_angle_deg, N, dt
):
    A = np.eye(6)
    A[0,3] = A[1,4] = A[2,5] = dt

    B = np.zeros((6,3))
    B[0:3,0:3] = 0.5 * dt**2 * np.eye(3)
    B[3:6,0:3] = dt * np.eye(3)

    n_app = docking_axis / np.linalg.norm(docking_axis)
    cos_th = np.cos(np.deg2rad(cone_angle_deg))

    # --- smart initialization ---
    p_entry = p_goal - 2.5 * n_app
    x_ref = np.zeros((6, N))
    split = int(0.6 * N)

    for k in range(N):
        if k < split:
            a = k / split
            x_ref[0:3,k] = (1-a)*p_start + a*p_entry
            x_ref[2,k] += 0.5*np.sin(np.pi*a)
            x_ref[3:6,k] = (1-a)*v_start
        else:
            a = (k-split)/(N-split)
            x_ref[0:3,k] = (1-a)*p_entry + a*p_goal

    history = {"X": [], "U": [], "cost": []}

    for _ in range(15):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        s_obs  = cp.Variable(N, nonneg=True)
        s_cone = cp.Variable(N, nonneg=True)

        cost = (
            0.1 * cp.sum_squares(u)
            + 1e4 * (cp.sum(s_obs) + cp.sum(s_cone))
        )

        cons = [x[:,0] == np.hstack([p_start, v_start])]

        for k in range(N-1):
            cons += [
                x[:,k+1] == A@x[:,k] + B@u[:,k],
                cp.norm(u[:,k]) <= 12.0
            ]

        cons += [
            cp.norm(x[0:3,-1] - p_goal) <= 0.05,
            cp.norm(x[3:6,-1]) <= 0.1
        ]

        for k in range(1,N):
            n = x_ref[0:3,k] - p_obs
            n /= np.linalg.norm(n) + 1e-6
            cons += [n @ (x[0:3,k] - p_obs) >= r_obs - s_obs[k]]

            if k > split:
                p_rel = x[0:3,k] - p_goal
                d = -n_app @ p_rel
                cons += [
                    d >= 0,
                    cp.norm(p_rel)*cos_th <= d + s_cone[k]
                ]

            cons += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.CLARABEL)

        if x.value is None:
            break

        history["X"].append(x.value.copy())
        history["U"].append(u.value.copy())
        history["cost"].append(prob.value)

        if np.linalg.norm(x.value - x_ref) < 0.1:
            break

        x_ref = x.value.copy()

    return history

# ======================================================================
# ARCHITECTURE 1: ACCEL → SETPOINT
# ======================================================================
def architecture1_accel(k, X, U, p, v):
    Kp = np.diag([2.0, 2.0, 2.5])
    Kd = np.diag([1.5, 1.5, 2.0])

    p_ref = X[0:3,k]
    v_ref = X[3:6,k]
    u_ff  = U[:,k] if k < U.shape[1] else np.zeros(3)

    return u_ff + Kp@(p_ref-p) + Kd@(v_ref-v)

def accel_to_setpoint(p, v, a, dt):
    return p + v*dt + 0.5*a*dt**2, v + a*dt

# ======================================================================
# VISUALIZATION HELPERS (UNCHANGED CONCEPTUALLY)
# ======================================================================
def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                              rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseVisualShapeIndex=vis,
                             basePosition=[0,0,0],
                             physicsClientId=client)

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos,
                                      [0,0,0,1],
                                      physicsClientId=client)

def visualize_docking_cone(p_goal, axis, angle_deg, length=2.0, client=0):
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis
    theta = np.deg2rad(angle_deg)

    ref = np.array([0,0,1]) if abs(axis[2]) < 0.9 else np.array([0,1,0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    for z in np.linspace(0, length, 15):
        r = z*np.tan(theta)
        c = p_goal + cone_dir*z
        prev = None
        for phi in np.linspace(0, 2*np.pi, 24):
            pt = c + r*(u*np.cos(phi) + v*np.sin(phi))
            if prev is not None:
                p.addUserDebugLine(prev, pt, [0.2,0.8,0.2], 1,
                                   physicsClientId=client)
            prev = pt

# ======================================================================
# MAIN
# ======================================================================
def run():

    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 1.5, 0.0, 0.6])
    OBS_POS      = np.array([-0.5, 0.1, 0.6])

    history = plan_scp_docking(
        CHASER_START, np.zeros(3),
        TARGET_POS, OBS_POS, 0.8,
        DOCKING_AXIS, CONE_ANGLE, N, DT
    )

    X = history["X"][-1]
    U = history["U"][-1]

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, TARGET_POS]),
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True
    )

    client = env.getPyBulletClient()

    hull_c = create_hull(SAFETY_R, [0,1,1,0.3], client)
    hull_t = create_hull(SAFETY_R, [1,0,1,0.3], client)

    visualize_docking_cone(TARGET_POS, DOCKING_AXIS, CONE_ANGLE, client=client)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((2,4))
    START = time.time()

    for k in range(X.shape[1]-1):

        obs,_,_,_,_ = env.step(action)

        p = obs[0][0:3]
        v = obs[0][10:13]

        update_hull(hull_c, p, client)
        update_hull(hull_t, obs[1][0:3], client)

        a = architecture1_accel(k, X, U, p, v)
        p_cmd, v_cmd = accel_to_setpoint(p, v, a, DT)

        action[0],_,_ = ctrl.computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], p_cmd, np.zeros(3), v_cmd
        )

        action[1],_,_ = ctrl.computeControlFromState(
            env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3), np.zeros(3)
        )

        env.render()
        sync(k, START, DT)

    env.close()

if __name__ == "__main__":
    run()
