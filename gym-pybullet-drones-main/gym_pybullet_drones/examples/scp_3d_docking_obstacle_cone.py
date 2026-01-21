import time
import numpy as np
import cvxpy as cp
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync


DOCKING_AXIS = np.array([0.0, 1.0, 0.0]) 
CONE_ANGLE   = 20  # Degrees
DURATION_SEC = 20  # Increased to allow time for the turn


def plan_scp_docking(
    p_start,
    p_goal,
    p_obs,
    r_obs,
    docking_axis, 
    cone_angle_deg,
    N,
    dt
):
    print(f"SCP: Planning for Axis {docking_axis}...")
    
    # Dynamics
    A = np.eye(6); A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
    B = np.zeros((6,3)); B[0,0]=0.5*dt**2; B[1,1]=0.5*dt**2; B[2,2]=0.5*dt**2; B[3,0]=dt; B[4,1]=dt; B[5,2]=dt

    # Normalize Axis
    n_approach = docking_axis / np.linalg.norm(docking_axis)
    cos_theta = np.cos(np.deg2rad(cone_angle_deg))

    # --- SMART INITIALIZATION ---
    # We define an "Entry Point" 2.5m away along the docking axis
    p_entry = p_goal - n_approach * 2.5
    
    x_ref = np.zeros((6, N))
    split_idx = int(0.6 * N) # 60% time to reach entry, 40% to dock
    
    for k in range(N):
        if k < split_idx:
            # Phase 1: Fly to Entry Point (Go around obstacle)
            alpha = k / split_idx
            x_ref[0:3, k] = (1 - alpha) * p_start + alpha * p_entry
            # Add Z-height clearance bump
            x_ref[2, k] += 0.8 * np.sin(np.pi * alpha) 
        else:
            # Phase 2: Fly Straight In (Entry -> Goal)
            alpha = (k - split_idx) / (N - split_idx)
            x_ref[0:3, k] = (1 - alpha) * p_entry + alpha * p_goal

    # --- OPTIMIZATION LOOP ---
    max_iters = 15
    for iteration in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack_obs = cp.Variable(N, nonneg=True)
        slack_cone = cp.Variable(N, nonneg=True)

        # Cost
        cost = cp.sum_squares(u) + 1000*cp.sum(slack_obs) + 1000*cp.sum(slack_cone)
        constraints = []

        # Boundary
        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3)])]
        
        # Dynamics
        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

        # Terminal
        constraints += [cp.norm(x[0:3,-1] - p_goal) <= 0.02]
        constraints += [cp.norm(x[3:6,-1]) <= 0.05]

        for k in range(1, N):
            # 1. Obstacle
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)
            n_obs_vec = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [n_obs_vec @ (x[0:3,k] - p_obs) >= r_obs - slack_obs[k]]

            # 2. Docking Cone (Enforced ONLY in Phase 2)
            if k > split_idx:
                p_rel = x[0:3,k] - p_goal 
                # Distance "along" the cone axis (should be positive away from goal)
                # We want p_rel to be in direction of -n_approach
                dist_long = -n_approach @ p_rel
                
                constraints += [dist_long >= 0]
                constraints += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]

        # Trust Region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Solver Fallback Logic
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            try:
                prob.solve(solver=cp.ECOS)
            except:
                prob.solve(solver=cp.SCS)

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
# 3. VISUALIZATION HELPERS
# ======================================================================
def draw_planned_path(traj, client):
    """Draws the mathematical path in BLUE lines"""
    for i in range(len(traj) - 1):
        p.addUserDebugLine(traj[i], traj[i+1], [0, 0, 1], 3, physicsClientId=client)

def create_spherical_obstacle(p_obs, r_obs, client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_obs, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r_obs, rgbaColor=[1,0,0,0.4], physicsClientId=client)
    p.createMultiBody(0, col, vis, p_obs, physicsClientId=client)

def visualize_docking_cone(p_goal, axis, angle_deg, length=2.0, client=0):
    axis = axis / np.linalg.norm(axis)
    # Cone opens in direction -axis
    cone_dir_main = -axis 
    theta = np.deg2rad(angle_deg)
    
    # Basis
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    # Draw rim
    for phi in np.linspace(0, 2*np.pi, 12):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir_main * np.cos(theta) + radial * np.sin(theta)
        end_pos = p_goal + vec * length
        p.addUserDebugLine(p_goal, end_pos, [0,1,0], 2, physicsClientId=client)
    
    # Draw Axis
    p.addUserDebugLine(p_goal, p_goal + cone_dir_main*length, [0,1,0], 4, physicsClientId=client)

# ======================================================================
# 4. MAIN EXECUTION
# ======================================================================
def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    
    # SETUP
    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 1.5, 0.0, 0.6]) 
    P_OBS        = np.array([-0.5, 0.1, 0.6]) 
    R_OBS        = 0.6

    INIT_XYZS = np.array([CHASER_START, TARGET_POS])
    INIT_RPYS = np.zeros((2,3))

    # PLAN
    traj = plan_scp_docking(
        p_start=CHASER_START,
        p_goal=TARGET_POS,
        p_obs=P_OBS,
        r_obs=R_OBS,
        docking_axis=DOCKING_AXIS,
        cone_angle_deg=CONE_ANGLE,
        N=int(CTRL_FREQ * DURATION_SEC),
        dt=1/CTRL_FREQ
    )

    # SIM ENV
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
    
    # ADD VISUALS
    create_spherical_obstacle(P_OBS, R_OBS, PYB_CLIENT)
    visualize_docking_cone(TARGET_POS, DOCKING_AXIS, CONE_ANGLE, client=PYB_CLIENT)
    
    # DRAW THE PLANNED PATH (DEBUGGING)
    draw_planned_path(traj, PYB_CLIENT)

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    # FLIGHT LOOP
    for i in range(len(traj)):
        obs, _, _, _, _ = env.step(action)

        # Chaser
        action[0], _, _ = ctrl[0].computeControlFromState(
            env.CTRL_TIMESTEP, obs[0], traj[i], np.zeros(3)
        )
        
        # Target
        action[1], _, _ = ctrl[1].computeControlFromState(
            env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3)
        )

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)

    # HOVER HOLD
    print("Trajectory Complete. Holding...")
    for _ in range(SIM_FREQ * 5): 
        obs, _, _, _, _ = env.step(action)
        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], TARGET_POS, np.zeros(3))
        action[1], _, _ = ctrl[1].computeControlFromState(env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3))
        env.render()
        time.sleep(1/SIM_FREQ)

    env.close()

if __name__ == "__main__":
    run()