import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from scp_planner_3d import plan_scp_trajectory

DEFAULT_DRONE = DroneModel.CF2X
DEFAULT_PHYSICS = Physics.PYB

def run():
    #### Simulation params #####################################
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION = 12

    #### Initial positions #####################################
    CHASER_START = np.array([-2.0, 0.0, 0.5])
    TARGET_POS   = np.array([ 0.0, 0.0, 0.5])

    INIT_XYZS = np.array([
        CHASER_START,
        TARGET_POS
    ])
    INIT_RPYS = np.zeros((2,3))

    #### Plan SCP trajectory ###################################
    traj = plan_scp_trajectory(
        p_start=CHASER_START,
        p_goal=TARGET_POS,
        N=int(CTRL_FREQ * DURATION),
        dt=1/CTRL_FREQ
    )

    #### Environment ###########################################
    env = CtrlAviary(
        drone_model=DEFAULT_DRONE,
        num_drones=2,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=DEFAULT_PHYSICS,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False
    )

    #### Controllers ###########################################
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONE) for _ in range(2)]

    #### Simulation loop #######################################
    action = np.zeros((2,4))
    START = time.time()

    for i in range(len(traj)):
        obs, _, _, _, _ = env.step(action)

        # --- Chaser drone (index 0) ---
        action[0], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=traj[i],
            target_rpy=np.zeros(3)
        )

        # --- Target drone (index 1) ---
        action[1], _, _ = ctrl[1].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[1],
            target_pos=TARGET_POS,
            target_rpy=np.zeros(3)
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()
