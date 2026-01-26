import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from scp_planner_3D_docking_collision import plan_scp_3d_pose_dcol


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(2*(w*y-z*x))
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([roll, pitch, yaw])


def run():
    p_start = np.array([-2.0, 0.0, 0.6])
    p_target = np.array([0.0, 0.0, 0.6])

    traj, quat_traj = plan_scp_3d_pose_dcol(p_start, p_target)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([p_start, p_target]),
        initial_rpys=np.zeros((2, 3)),
        physics=Physics.PYB,
        gui=True,
        ctrl_freq=48,
        pyb_freq=240
    )

    client = env.getPyBulletClient()

    # Translucent collision hulls
    hull_radius = 0.25
    hull_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=hull_radius,
        rgbaColor=[0, 0, 1, 0.25],
        physicsClientId=client
    )

    hull_ids = [
        p.createMultiBody(0, -1, hull_visual, p_start, physicsClientId=client),
        p.createMultiBody(0, -1, hull_visual, p_target, physicsClientId=client)
    ]

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2, 4))

    START = time.time()

    for i in range(traj.shape[1]):
        obs, _, _, _, _ = env.step(action)

        # Update hulls
        p.resetBasePositionAndOrientation(
            hull_ids[0],
            traj[0:3, i],
            [0, 0, 0, 1],
            physicsClientId=client
        )

        # Chaser control
        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[0],
            traj[0:3, i],
            quat_to_rpy(quat_traj[:, i])
        )

        # Target fixed
        action[1], _, _ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[1],
            p_target,
            np.zeros(3)
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    env.close()


if __name__ == "__main__":
    run()
