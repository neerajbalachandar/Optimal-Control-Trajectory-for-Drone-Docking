import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from scp_docking_dcol_planner import (
    solve_initial_docking,
    solve_scp_dcol,
    alpha_dcol
)

# ----------------------------
# PARAMETERS
# ----------------------------
R_C = 0.25
R_T = 0.25
ALPHA_WARN = 1.1

# ----------------------------
def run():
    p0 = np.array([-2.5,0,0.6])
    p_target = np.array([0,0,0.6])
    axis = np.array([1,0,0])

    N = 120
    dt = 1/48

    # Problem 2
    x_init = solve_initial_docking(p0, p_target, axis, 20, N, dt)

    # Problem 3
    x_plan = solve_scp_dcol(x_init, p_target, R_C, R_T, N, dt)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([p0,p_target]),
        initial_rpys=np.zeros((2,3)),
        physics=Physics.PYB,
        gui=True,
        ctrl_freq=48,
        pyb_freq=240
    )

    client = env.getPyBulletClient()

    hull_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=R_C,
        rgbaColor=[0,0,1,0.3],
        physicsClientId=client
    )
    hull_c = p.createMultiBody(0,-1,hull_vis,p0)

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    for k in range(x_plan.shape[1]):
        obs,_,_,_,_ = env.step(action)

        p_c = obs[0][0:3]
        alpha = alpha_dcol(p_c, p_target, R_C, R_T)

        # Runtime α-supervisor
        if alpha < ALPHA_WARN:
            print(f"α warning {alpha:.2f} → replanning")
            x_init = solve_initial_docking(p_c, p_target, axis, 20, N, dt)
            x_plan = solve_scp_dcol(x_init, p_target, R_C, R_T, N, dt)
            k = 0
            continue

        p.resetBasePositionAndOrientation(hull_c, p_c, [0,0,0,1])

        action[0],_,_ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], x_plan[0:3,k], np.zeros(3)
        )
        action[1],_,_ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP, obs[1], p_target, np.zeros(3)
        )

        env.render()
        sync(k, START, env.CTRL_TIMESTEP)

        # docking convergence
        if np.linalg.norm(p_c - p_target) < 0.03 and alpha >= 1.0:
            print("Docking complete and safe.")
            break

    env.close()

if __name__ == "__main__":
    run()
