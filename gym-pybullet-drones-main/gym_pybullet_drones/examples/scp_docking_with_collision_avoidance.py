import time
import numpy as np
import cvxpy as cp
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ============================================================
# PARAMETERS
# ============================================================
SIM_FREQ  = 240
CTRL_FREQ = 48
DT        = 1 / CTRL_FREQ

DURATION_SEC = 20
N = int(DURATION_SEC * CTRL_FREQ)

DOCKING_AXIS = np.array([0.0, 1.0, 0.0])
CONE_ANGLE   = np.deg2rad(20.0)

CHASER_HE = np.array([0.125, 0.125, 0.04])
TARGET_HE = np.array([0.175, 0.175, 0.06])

# ============================================================
# SCP PLANNER (ROBUST)
# ============================================================
def plan_scp_docking(p_start, p_goal):
    print("[SCP] Planning trajectory...")

    A = np.eye(6)
    A[0,3] = DT; A[1,4] = DT; A[2,5] = DT
    B = np.zeros((6,3))
    B[0:3,:] = 0.5 * DT**2 * np.eye(3)
    B[3:6,:] = DT * np.eye(3)

    n_axis = DOCKING_AXIS / np.linalg.norm(DOCKING_AXIS)
    cos_th = np.cos(CONE_ANGLE)

    p_entry = p_goal - 2.5 * n_axis

    x_ref = np.zeros((6, N))
    split = int(0.6 * N)

    for k in range(N):
        if k < split:
            a = k / split
            x_ref[0:3,k] = (1-a)*p_start + a*p_entry
            x_ref[2,k] += 0.8 * np.sin(np.pi * a)
        else:
            a = (k - split) / (N - split)
            x_ref[0:3,k] = (1-a)*p_entry + a*p_goal

    for it in range(8):
        x = cp.Variable((6, N))
        u = cp.Variable((3, N-1))
        slack = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + 500 * cp.sum(slack)
        cons = [x[:,0] == np.hstack([p_start, np.zeros(3)])]

        for k in range(N-1):
            cons += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]

        cons += [
            cp.norm(x[0:3,-1] - p_goal) <= 0.05,
            cp.norm(x[3:6,-1]) <= 0.1
        ]

        for k in range(split, N):
            p_rel = x[0:3,k] - p_goal
            dlong = -n_axis @ p_rel
            cons += [
                dlong >= 0,
                cp.norm(p_rel) * cos_th <= dlong + slack[k]
            ]

        for k in range(N):
            cons += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), cons)

        solved = False
        for solver in [cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True)
                if x.value is not None:
                    solved = True
                    break
            except:
                pass

        if not solved:
            print("[SCP] Solver failed at iteration", it)
            break

        x_ref = x.value.copy()

    traj = x_ref[0:3,:].T
    print("[SCP] Trajectory length:", len(traj))
    return traj

# ============================================================
# VISUALS
# ============================================================
def draw_docking_cone(p_goal, axis, client):
    axis = axis / np.linalg.norm(axis)
    for a in np.linspace(-CONE_ANGLE, CONE_ANGLE, 12):
        d = -axis*np.cos(a) + np.array([0,0,1])*np.sin(a)
        p.addUserDebugLine(p_goal, p_goal + 2*d, [0,1,0], 2, physicsClientId=client)

# ============================================================
# MAIN
# ============================================================
def run():
    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 1.5, 0.0, 0.6])

    traj = plan_scp_docking(CHASER_START, TARGET_POS)
    if traj is None or len(traj) < 5:
        raise RuntimeError("Trajectory generation failed")

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, TARGET_POS]),
        initial_rpys=np.zeros((2,3)),
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False
    )

    client = env.getPyBulletClient()
    ctrl = [DSLPIDControl(DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))

    # Hulls (created once)
    chaser_hull = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=CHASER_HE),
        p.createVisualShape(p.GEOM_BOX, halfExtents=CHASER_HE, rgbaColor=[0,0,1,0.25]),
        CHASER_START
    )

    target_hull = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=TARGET_HE),
        p.createVisualShape(p.GEOM_BOX, halfExtents=TARGET_HE, rgbaColor=[1,0,0,0.25]),
        TARGET_POS
    )

    draw_docking_cone(TARGET_POS, DOCKING_AXIS, client)

    START = time.time()

    for i in range(len(traj)):
        obs, _, _, _, _ = env.step(action)

        yaw = obs[0][9]
        desired_rpy = np.array([0.0, 0.0, yaw])

        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], traj[i], np.zeros(3),
            desired_rpy=desired_rpy
        )

        action[1], _, _ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3)
        )

        p.resetBasePositionAndOrientation(
            chaser_hull, obs[0][0:3], obs[0][3:7]
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    print("[INFO] Trajectory complete. Holding hover...")
    for _ in range(5 * SIM_FREQ):
        obs, _, _, _, _ = env.step(action)
        env.render()
        time.sleep(1 / SIM_FREQ)

    env.close()

if __name__ == "__main__":
    run()
