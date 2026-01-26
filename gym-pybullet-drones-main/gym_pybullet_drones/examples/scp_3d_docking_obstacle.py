import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from scp_3d_planner_obstacle import plan_scp_trajectory

def create_spherical_obstacle(p_obs, r_obs, client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_obs, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r_obs,
                              rgbaColor=[1,0,0,0.4], physicsClientId=client)
    p.createMultiBody(0, col, vis, p_obs, physicsClientId=client)

def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION = 14

    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 0.5, 0.0, 0.6])

    # Obstacle
    P_OBS = np.array([-1.0, 0.0, 0.6])
    R_OBS = 0.5

    INIT_XYZS = np.array([CHASER_START, TARGET_POS])
    INIT_RPYS = np.zeros((2,3))

    traj = plan_scp_trajectory(
        p_start=CHASER_START,
        p_goal=TARGET_POS,
        p_obs=P_OBS,
        r_obs=R_OBS,
        N=int(CTRL_FREQ * DURATION),
        dt=1/CTRL_FREQ
    )

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False
    )

    PYB_CLIENT = env.getPyBulletClient()
    create_spherical_obstacle(P_OBS, R_OBS, PYB_CLIENT)

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    for i in range(len(traj)):
        obs, _, _, _, _ = env.step(action)

        # Chaser
        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[0],
            traj[i],
            np.zeros(3)
        )

        # Target (hover)
        action[1], _, _ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[1],
            TARGET_POS,
            np.zeros(3)
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()
