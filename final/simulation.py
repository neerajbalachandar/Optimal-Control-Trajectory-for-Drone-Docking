import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.final.dcol import solve_dcol_scaling, create_hull, update_hull
from gym_pybullet_drones.final.target_prediction_linear import LinearTarget, TargetEKF
from gym_pybullet_drones.final.target_prediction_linear import AsyncPlanner
from gym_pybullet_drones.final.scp_planner import *

# ============================================================
# CONFIG
# ============================================================
SIM_FREQ   = 240
CTRL_FREQ  = 48
DT         = 1 / CTRL_FREQ
DURATION   = 25.0

HORIZON        = 25
DT_PLAN        = DT
PLAN_INTERVAL  = 0.4

SAFETY_R       = 0.1
ALPHA_LIMIT    = 1.05

# Docking cone (target body frame)
DOCKING_AXIS = np.array([0.0, 1.0, 0.0])
CONE_ANGLE   = 20.0        # degrees
CONE_LENGTH  = 2.0

# Obstacle (used by SCP)
P_OBS = np.array([-1.5, 0.0, 1.0])
R_OBS = 0.4


# ============================================================
def run():

    # --------------------------------------------------------
    # INITIAL CONDITIONS
    # --------------------------------------------------------
    CHASER_START = np.array([-2.5, 0.0, 1.2])

    target_model = LinearTarget()
    ekf = TargetEKF(dt=DT)
    ekf.x = target_model.get_state(0.0)

    planner = AsyncPlanner(
        N=HORIZON,
        dt=DT_PLAN,
        p_obs=np.array([-1.5, 0.0, 1.0]),
        r_obs=0.4,
        docking_axis=DOCKING_AXIS,
        cone_angle_deg=CONE_ANGLE
    )

    planner.start()

    # --------------------------------------------------------
    # ENVIRONMENT (DOWNWASH ENABLED)
    # --------------------------------------------------------
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([
            CHASER_START,
            ekf.x[0:3]
        ]),
        physics=Physics.PYB_DW,
        neighbourhood_radius=5.0,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True
    )

    PYB = env.getPyBulletClient()

    # --------------------------------------------------------
    # VISUALIZE OBSTACLE (USED BY SCP)
    # --------------------------------------------------------
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=R_OBS, physicsClientId=PYB)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=R_OBS,
                              rgbaColor=[1, 0, 0, 0.4], physicsClientId=PYB)
    p.createMultiBody(0, col, vis, P_OBS, physicsClientId=PYB)

    # --------------------------------------------------------
    # CONTROLLERS
    # --------------------------------------------------------
    ctrl = [
        DSLPIDControl(drone_model=DroneModel.CF2X),
        DSLPIDControl(drone_model=DroneModel.CF2X)
    ]
    action = np.zeros((2, 4))

    # --------------------------------------------------------
    # VISUAL HULLS (DCOL)
    # --------------------------------------------------------
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.4], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.4], PYB)

    traj_ids = []
    cone_ids = []

    curr_traj = None
    last_plan = -np.inf
    safe_mode = False

    START = time.time()

    # ========================================================
    # MAIN LOOP
    # ========================================================
    for i in range(int(DURATION * CTRL_FREQ)):

        sim_t = i / CTRL_FREQ

        # ------------------- STEP PHYSICS --------------------
        obs, _, _, _, _ = env.step(action)
        pos_c = obs[0][0:3]
        vel_c = obs[0][10:13]

        # ------------------- TARGET MOTION -------------------
        true_state = target_model.get_state(sim_t)
        true_state[2] = max(true_state[2], 0.1)

        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[1],
            true_state[0:3],
            [0, 0, 0, 1],
            physicsClientId=PYB
        )

        # ------------------- EKF -----------------------------
        ekf.step(true_state[0:3] + np.random.normal(0, 0.02, 3))

        # ------------------- UPDATE MOVING CONE --------------
        for uid in cone_ids:
            p.removeUserDebugItem(uid, physicsClientId=PYB)
        cone_ids.clear()

        cone_ids = draw_moving_docking_cone(
            p_goal=true_state[0:3],
            axis=DOCKING_AXIS,
            angle_deg=CONE_ANGLE,
            length=CONE_LENGTH,
            client=PYB
        )

        # ------------------- PLANNER RESULT ------------------
        res = planner.get()
        if res is not None:
            curr_traj = res

            for uid in traj_ids:
                p.removeUserDebugItem(uid, physicsClientId=PYB)
            traj_ids.clear()

            for k in range(curr_traj.shape[0] - 1):
                p.addUserDebugLine(
                    curr_traj[k],
                    curr_traj[k+1],
                    [0, 0, 1], 3,
                    physicsClientId=PYB
                )

        # ------------------- PLANNER REQUEST -----------------
        if sim_t - last_plan > PLAN_INTERVAL:
            preds = ekf.predict_future(HORIZON, DT_PLAN)
            planner.request(
                np.hstack([pos_c, vel_c]),
                preds
            )
            last_plan = sim_t

        # ------------------- DCOL CHECK ----------------------
        alpha, contact = solve_dcol_scaling(
            pos_c, SAFETY_R,
            true_state[0:3], SAFETY_R
        )

        if alpha < ALPHA_LIMIT:
            safe_mode = True
            p.addUserDebugText("DCOL", contact, [1, 0, 0], 2,
                               physicsClientId=PYB)

        # ------------------- CONTROL -------------------------
        if curr_traj is not None and not safe_mode:
            rel = curr_traj[0:3, 0] - true_state[0:3]
            if np.linalg.norm(rel) < 1.0:
                pt = curr_traj[0:3, 0]
                vt = curr_traj[3:6, 0]
            else:
                pt = curr_traj[0:3, 2]
                vt = curr_traj[3:6, 2]
        else:
            pt = pos_c
            vt = np.zeros(3)

        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], pt, vt
        )
        action[1] = np.zeros(4)

        # ------------------- VISUAL UPDATE -------------------
        update_hull(hull_c, pos_c, PYB)
        update_hull(hull_t, true_state[0:3], PYB)

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    env.close()


if __name__ == "__main__":
    run()
