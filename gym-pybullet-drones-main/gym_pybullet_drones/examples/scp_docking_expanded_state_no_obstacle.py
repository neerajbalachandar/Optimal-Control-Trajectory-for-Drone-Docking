import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from scp_planner_expanded_state_no_obstacle import plan_scp_13d

# ----------------------------
# Quaternion to RPY
# ----------------------------
def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(2*(qw*qy - qz*qx))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return np.array([roll, pitch, yaw])

# ----------------------------
# Main simulation
# ----------------------------
def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION = 14

    # Initial / target states
    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 0.5, 0.0, 0.6])

    q_start = np.array([1., 0., 0., 0.])
    q_goal  = np.array([1., 0., 0., 0.])

    # Obstacle
    P_OBS = np.array([-1.0, 0.0, 0.6])
    R_OBS = 0.5

    INIT_XYZS = np.array([CHASER_START, TARGET_POS])
    INIT_RPYS = np.zeros((2,3))

    # ----------------------------
    # SCP planning
    # ----------------------------
    X = plan_scp_13d(
        p_start=CHASER_START,
        q_start=q_start,
        p_goal=TARGET_POS,
        q_goal=q_goal,
        p_obs=P_OBS,
        r_obs=R_OBS,
        N=int(CTRL_FREQ * DURATION),
        dt=1/CTRL_FREQ
    )

    pos_traj = X[0:3, :].T
    quat_traj = X[6:10, :].T

    # ----------------------------
    # Environment
    # ----------------------------
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

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    # ----------------------------
    # Simulation loop
    # ----------------------------
    for i in range(pos_traj.shape[0]):
        obs, _, _, _, _ = env.step(action)

        # Chaser
        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[0],
            pos_traj[i],
            quat_to_rpy(quat_traj[i])
        )

        # Target (static)
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
