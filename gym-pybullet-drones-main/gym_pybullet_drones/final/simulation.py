# unified_docking_simulation.py

import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.final.dcol import *
from gym_pybullet_drones.final.target_prediction import *
from gym_pybullet_drones.final.scp_planner import *
from gym_pybullet_drones.final.downwash_model import * 


SIM_FREQ   = 240
CTRL_FREQ  = 48
DT         = 1 / CTRL_FREQ
DURATION   = 25.0

HORIZON        = 25
DT_PLAN        = 1 / CTRL_FREQ
PLAN_INTERVAL  = 0.4

SAFETY_R       = 0.35
ALPHA_LIMIT    = 1.05


def run():

    # --------------------------------------------------------
    # INITIAL CONDITIONS
    # --------------------------------------------------------
    CHASER_START = np.array([-2.5, 0.0, 1.2])

    target_model = UnstableHoverTarget()
    ekf = TargetEKF(dt=DT)
    ekf.x = target_model.get_state(0.0)

    planner = AsyncPlanner(
        N=HORIZON,
        dt=DT_PLAN,
        p_obs=np.array([-1.5, 0.0, 1.0]),
        r_obs=0.4
    )
    planner.start()

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([
            CHASER_START,
            ekf.x[0:3]
        ]),
        physics=Physics.PYB_DW,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False
    )

    PYB = env.getPyBulletClient()

    # --------------------------------------------------------
    # CONTROLLERS
    # --------------------------------------------------------
    ctrl = [
        DSLPIDControl(drone_model=DroneModel.CF2X),
        DSLPIDControl(drone_model=DroneModel.CF2X)
    ]

    action = np.zeros((2, 4))

    # --------------------------------------------------------
    # VISUALS
    # --------------------------------------------------------
    hull_c = create_hull(SAFETY_R, [0,1,1,0.4], PYB)
    hull_t = create_hull(SAFETY_R, [1,0,1,0.4], PYB)

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------
    curr_traj = None
    last_plan = -np.inf
    safe_mode = False

    START = time.time()

    
    for i in range(int(DURATION * CTRL_FREQ)):

        sim_t = i / CTRL_FREQ

        # ----------------------------------------------------
        # 1. STEP PHYSICS
        # ----------------------------------------------------
        obs, _, _, _, _ = env.step(action)

        pos_c = obs[0][0:3]
        vel_c = obs[0][10:13]

        # ----------------------------------------------------
        # 2. TRUE TARGET MOTION
        # ----------------------------------------------------
        true_state = target_model.get_state(sim_t)
        true_state[2] = max(true_state[2], 0.1)

        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[1],
            true_state[0:3],
            [0,0,0,1],
            physicsClientId=PYB
        )

        # ----------------------------------------------------
        # 3. EKF UPDATE
        # ----------------------------------------------------
        z = true_state[0:3] + np.random.normal(0, 0.02, 3)
        ekf.step(z)

        # ----------------------------------------------------
        # 4. PLANNER RESPONSE
        # ----------------------------------------------------
        res = planner.get()
        if res is not None:
            curr_traj = res
            p.removeAllUserDebugItems(PYB)
            for k in range(HORIZON-1):
                p.addUserDebugLine(
                    curr_traj[0:3,k],
                    curr_traj[0:3,k+1],
                    [0,0,1], 2, PYB
                )

        # ----------------------------------------------------
        # 5. PLANNER REQUEST
        # ----------------------------------------------------
        if sim_t - last_plan > PLAN_INTERVAL:
            preds = ekf.predict_future(HORIZON, DT_PLAN)
            chaser_state = np.hstack([pos_c, vel_c])
            planner.request(chaser_state, preds)
            last_plan = sim_t

        # ----------------------------------------------------
        # 6. DCOL SAFETY CHECK
        # ----------------------------------------------------
        alpha, contact = solve_dcol_scaling(
            pos_c, SAFETY_R,
            true_state[0:3], SAFETY_R
        )

        if alpha < ALPHA_LIMIT:
            safe_mode = True
            p.addUserDebugText("DCOL TRIGGER", contact, [1,0,0], 2, PYB)

        # ----------------------------------------------------
        # 7. CONTROL SELECTION
        # ----------------------------------------------------
        if curr_traj is not None and not safe_mode:
            pt = curr_traj[0:3, min(2, HORIZON-1)]
            vt = curr_traj[3:6, min(2, HORIZON-1)]
        else:
            pt = pos_c
            vt = np.zeros(3)

        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], pt, vt
        )

        action[1] = np.zeros(4)

        # ----------------------------------------------------
        # 8. VISUAL UPDATE
        # ----------------------------------------------------
        update_hull(hull_c, pos_c, PYB)
        update_hull(hull_t, true_state[0:3], PYB)

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    env.close()


if __name__ == "__main__":
    run()
