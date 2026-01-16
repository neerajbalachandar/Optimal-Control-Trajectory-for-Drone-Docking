"""
2D Circular Docking Simulation
- Target: Moves in a circle (Radius=1m) at constant height.
- Chaser: Uses PID to track a docking point 0.4m behind the target.
"""
import time
import argparse
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# --- Configuration ---
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2 
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_DURATION_SEC = 15
DEFAULT_CONTROL_FREQ_HZ = 48

def run(drone=DEFAULT_DRONES, num_drones=DEFAULT_NUM_DRONES, physics=DEFAULT_PHYSICS, 
        gui=DEFAULT_GUI, plot=DEFAULT_PLOT, duration_sec=DEFAULT_DURATION_SEC, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, **kwargs):

    # 1. Setup Environment
    # Target starts at [1, 0, 1], Chaser starts nearby
    INIT_XYZS = np.array([[1.0, 0, 1.0], [0.5, 0, 1.0]])
    INIT_RPYS = np.array([[0, 0, 0], [0, 0, 0]])

    env = CtrlAviary(drone_model=drone, num_drones=num_drones, initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS, physics=physics, ctrl_freq=control_freq_hz,
                     gui=gui, record=False, obstacles=False)
    
    # 2. Controllers & Logger
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones)

    # 3. Simulation Loop
    num_steps = int(duration_sec * env.CTRL_FREQ)
    action = np.zeros((num_drones, 4))
    START = time.time()
    
    # Trajectory Parameters
    RADIUS = 1.0
    ANGULAR_SPEED = 0.5 # rad/s

    for i in range(num_steps):
        t = i / env.CTRL_FREQ
        
        # --- A. Generate Simple 2D Circular Path ---
        # Calculate Target Position
        target_x = RADIUS * np.cos(ANGULAR_SPEED * t)
        target_y = RADIUS * np.sin(ANGULAR_SPEED * t)
        target_z = 1.0
        
        # Calculate Target Yaw (Tangent to circle) + pi/2 to face forward
        target_yaw = (ANGULAR_SPEED * t) + (np.pi / 2)
        
        target_pos = np.array([target_x, target_y, target_z])
        target_rpy = np.array([0, 0, target_yaw])

        # --- B. Force Target State (Perfect movement) ---
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0], 
            target_pos, 
            p.getQuaternionFromEuler(target_rpy)
        )
        
        # Step the physics environment
        obs, reward, terminated, truncated, info = env.step(action)

        # --- C. Chaser Logic (Simple Geometric Tracking) ---
        # The Chaser needs to be 0.4m BEHIND the target.
        # "Behind" means opposite to the velocity vector.
        
        # 1. Get Target Orientation Matrix
        target_quat = p.getQuaternionFromEuler(target_rpy)
        target_rot_mat = np.array(p.getMatrixFromQuaternion(target_quat)).reshape(3,3)
        
        # 2. Define Offset in Body Frame (Negative Y because standard PyBullet drone faces Y?)
        # Actually standard frame usually faces X. Let's assume X is forward.
        # -0.4 in X means 40cm behind.
        offset_body = np.array([-0.4, 0, 0]) 
        
        # 3. Transform to World Frame
        offset_world = target_rot_mat @ offset_body
        
        # 4. Calculate Goal
        chaser_goal_pos = target_pos + offset_world
        
        # Chaser should also face the same way as target
        chaser_goal_yaw = target_yaw

        # Calculate Control for Chaser
        action[1, :], _, _ = ctrl[1].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[1],
            target_pos=chaser_goal_pos,
            target_rpy=np.array([0, 0, chaser_goal_yaw])
        )
        
        # --- D. Visuals ---
        if gui and i % 5 == 0:
            p.removeAllUserDebugItems()
            # Draw line from Chaser to Target
            p.addUserDebugLine(obs[1][0:3], target_pos, [0, 0, 1], 2)
            # Draw Goal Point (Green)
            p.addUserDebugLine(chaser_goal_pos, chaser_goal_pos + [0,0,0.2], [0,1,0], 3)

        # Logging
        logger.log(drone=0, timestamp=t, state=obs[0], control=np.zeros(12))
        logger.log(drone=1, timestamp=t, state=obs[1], control=np.zeros(12))

        env.render()
        if gui: sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    if plot: logger.plot()

if __name__ == "__main__":
    run()
        
