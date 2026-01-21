import time
import numpy as np
import cvxpy as cp
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ======================================================================
# 1. GENERATE TARGET TRAJECTORY (PREDICTION)
# ======================================================================
def generate_target_trajectory(duration, freq):
    num_steps = int(duration * freq)
    dt = 1 / freq
    t = np.linspace(0, duration, num_steps)
    
    traj = np.zeros((num_steps, 6)) # [x, y, z, vx, vy, vz]
    
    # Motion: Drifting +X while bobbing in Y
    traj[:, 0] = 0.5 + 0.15 * t         # Moving forward
    traj[:, 1] = 0.8 * np.sin(0.8 * t)  # Bobbing left/right
    traj[:, 2] = 0.6                    # Constant Height
    
    # Velocities
    traj[1:, 3] = (traj[1:, 0] - traj[:-1, 0]) / dt
    traj[1:, 4] = (traj[1:, 1] - traj[:-1, 1]) / dt
    traj[1:, 5] = (traj[1:, 2] - traj[:-1, 2]) / dt
    
    return traj

# ======================================================================
# 2. SCP PLANNER (WITH DOCKING OFFSET)
# ======================================================================
def plan_scp_intercept(
    p_start,
    target_traj_future, 
    p_obs,
    r_obs,
    N,
    dt,
    docking_offset=np.array([-0.3, 0.0, 0.0]) # 30cm BEHIND target
):
    print("SCP: Planning Intercept with Safety Offset...")
    
    # Calculate the ACTUAL goal point (Target CoM + Offset)
    # The target is at target_traj_future[-1]
    # We want to be at target_pos + offset
    p_target_final = target_traj_future[-1, 0:3]
    v_target_final = target_traj_future[-1, 3:6]
    
    p_docking_port = p_target_final + docking_offset
    
    # Dynamics
    A = np.eye(6); A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
    B = np.zeros((6,3)); B[0,0]=0.5*dt**2; B[1,1]=0.5*dt**2; B[2,2]=0.5*dt**2; B[3,0]=dt; B[4,1]=dt; B[5,2]=dt

    # Initial Guess (Lead the target)
    x_ref = np.zeros((6, N))
    for k in range(N):
        alpha = k/(N-1)
        x_ref[0:3,k] = (1-alpha)*p_start + alpha*p_docking_port
        x_ref[2,k] += 1.0 * np.sin(np.pi*alpha) # Vertical bump for obstacle

    max_iters = 15
    for iteration in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + 1000*cp.sum(slack)
        constraints = []

        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3)])]
        
        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

        # --- TERMINAL CONSTRAINT ---
        # 1. Match Position (at the OFFSET point, not the center)
        constraints += [cp.norm(x[0:3,-1] - p_docking_port) <= 0.02]
        
        # 2. Match Velocity (Exact match to target speed)
        # This ensures they are "frozen" relative to each other
        constraints += [cp.norm(x[3:6,-1] - v_target_final) <= 0.05] 

        # Obstacle Avoidance
        for k in range(1, N):
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)
            n = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [n @ (x[0:3,k] - p_obs) >= r_obs - slack[k]]

        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            try: prob.solve(solver=cp.ECOS)
            except: prob.solve(solver=cp.SCS)

        if x.value is None:
            print(f"Solver Failed iter {iteration}")
            break
            
        diff = np.linalg.norm(x.value - x_ref)
        x_ref = x.value.copy()
        
        if diff < 0.1:
            print(f"Converged iter {iteration}")
            break

    return x_ref[0:3,:].T

# ======================================================================
# 3. VISUALIZATION & RUN
# ======================================================================
def create_spherical_obstacle(p_obs, r_obs, client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_obs, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r_obs, rgbaColor=[1,0,0,0.4], physicsClientId=client)
    p.createMultiBody(0, col, vis, p_obs, physicsClientId=client)

def draw_trajectory(traj, color, client):
    for i in range(len(traj)-1):
        p.addUserDebugLine(traj[i, 0:3], traj[i+1, 0:3], color, 2, physicsClientId=client)

def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION = 12

    CHASER_START = np.array([-2.5, 0.0, 0.6])
    P_OBS        = np.array([-0.5, 0.1, 0.6]) 
    R_OBS        = 0.5
    
    # OFFSET: Capture 30cm behind the target's center
    DOCKING_OFFSET = np.array([-0.3, 0.0, 0.0]) 

    # 1. Generate Target Future
    target_full_traj = generate_target_trajectory(DURATION, CTRL_FREQ)
    
    # 2. Plan Intercept
    chaser_traj = plan_scp_intercept(
        p_start=CHASER_START,
        target_traj_future=target_full_traj,
        p_obs=P_OBS,
        r_obs=R_OBS,
        N=len(target_full_traj),
        dt=1/CTRL_FREQ,
        docking_offset=DOCKING_OFFSET
    )

    # 3. Simulation
    INIT_XYZS = np.array([CHASER_START, target_full_traj[0, 0:3]])
    INIT_RPYS = np.zeros((2,3))

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
    
    PYB = env.getPyBulletClient()
    create_spherical_obstacle(P_OBS, R_OBS, PYB)
    draw_trajectory(target_full_traj, [1,0,0], PYB) 
    draw_trajectory(chaser_traj, [0,0,1], PYB)      

    # --- CRITICAL FIX: DISABLE COLLISION BETWEEN DRONES ---
    # This simulates the "Capture Mechanism" working instead of crashing
    # Body IDs are typically 0 (Obstacle), 1 (Drone 0), 2 (Drone 1) depending on load order
    # In CtrlAviary, drones are loaded first.
    # Drone 0 ID = 0, Drone 1 ID = 1. Obstacle ID = 2.
    # We disable collision between 0 and 1.
    p.setCollisionFilterPair(0, 1, -1, -1, 0, physicsClientId=PYB)

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    for i in range(len(chaser_traj)):
        obs, _, _, _, _ = env.step(action)

        # Chaser Control
        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], chaser_traj[i], np.zeros(3)
        )
        
        # Target Control
        target_pos_ref = target_full_traj[i, 0:3]
        target_vel_ref = target_full_traj[i, 3:6]
        action[1], _, _ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP, obs[1], target_pos_ref, target_vel_ref
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
        
    print("Soft Docking Complete. Formation Flying...")
    
    # HOLD LOOP: Formation Flying
    # Chaser maintains position RELATIVE to Target
    for _ in range(SIM_FREQ * 4):
        obs, _, _, _, _ = env.step(action)
        
        # Current Target State
        p_t = obs[1][0:3]
        v_t = obs[1][10:13]
        
        # Chaser Goal = Target + Offset
        p_c_goal = p_t + DOCKING_OFFSET
        
        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_c_goal, v_t)
        action[1], _, _ = ctrl[1].computeControlFromState(env.CTRL_TIMESTEP, obs[1], p_t + v_t*0.01, v_t) # Keep drifting
        
        env.render()
        time.sleep(1/SIM_FREQ)

    env.close()

if __name__ == "__main__":
    run()