import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from scp_13d_dcol_planner import plan_scp_13d_dcol

# =============================
# Convex hull definition
# =============================

def box_vertices(h):
    x,y,z = h
    return np.array([
        [ x, y, z], [ x,-y, z], [-x,-y, z], [-x, y, z],
        [ x, y,-z], [ x,-y,-z], [-x,-y,-z], [-x, y,-z]
    ])

# =============================
# Simulation
# =============================

def run():

    p_start = np.array([-1.5, 0, 0.6])
    q_start = np.array([1,0,0,0])
    x0 = np.hstack([p_start, np.zeros(3), q_start, np.zeros(3)])

    p_target = np.array([0,0,0.6])
    q_target = np.array([1,0,0,0])

    Vc = box_vertices([0.15,0.15,0.05])
    Vt = box_vertices([0.15,0.15,0.05])

    traj = plan_scp_13d_dcol(
        x0, p_target, q_target, Vc, Vt
    )

    # Interpolate for slow sim
    T = traj.shape[1]
    t_dense = np.linspace(0,1,10*T)
    traj_dense = np.vstack([
        np.interp(t_dense, np.linspace(0,1,T), traj[i])
        for i in range(traj.shape[0])
    ])

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([p_start,p_target]),
        initial_rpys=np.zeros((2,3)),
        physics=Physics.PYB,
        gui=True
    )

    client = env.getPyBulletClient()

    # Visual hulls (non-colliding)
    hull_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.15,0.15,0.05],
        rgbaColor=[0,0,1,0.25],
        physicsClientId=client
    )

    hull_ids = [
        p.createMultiBody(0,-1,hull_vis,[0,0,0]),
        p.createMultiBody(0,-1,hull_vis,[0,0,0])
    ]

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    for k in range(traj_dense.shape[1]):

        obs,_,_,_,_ = env.step(action)

        for i in range(2):
            pos, orn = p.getBasePositionAndOrientation(env.DRONE_IDS[i])
            p.resetBasePositionAndOrientation(
                hull_ids[i], pos, orn, physicsClientId=client
            )

        action[0],_,_ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[0],
            traj_dense[0:3,k],
            np.zeros(3)
        )

        action[1],_,_ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP,
            obs[1],
            p_target,
            np.zeros(3)
        )

        env.render()
        sync(k, START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()
