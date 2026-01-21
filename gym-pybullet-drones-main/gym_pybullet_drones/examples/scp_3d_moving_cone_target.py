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
# 1. GENERATE TARGET TRAJECTORY (Figure-8 Drift)
# ======================================================================
def generate_target_trajectory(duration, freq):
    num_steps = int(duration * freq)
    dt = 1 / freq
    t = np.linspace(0, duration, num_steps)
    
    traj = np.zeros((num_steps, 6))
    
    # Target moves slowly from X=0.5 to X=2.5
    traj[:, 0] = 0.5 + 0.08 * t          
    traj[:, 1] = 1.0 * np.sin(0.4 * t)  
    traj[:, 2] = 0.6 + 0.2 * np.cos(0.4 * t) # Slight vertical bob
    
    # Velocities
    traj[1:, 3] = (traj[1:, 0] - traj[:-1, 0]) / dt
    traj[1:, 4] = (traj[1:, 1] - traj[:-1, 1]) / dt
    traj[1:, 5] = (traj[1:, 2] - traj[:-1, 2]) / dt
    
    return traj

# ======================================================================
# 2. ROBUST SCP PLANNER (Handles Fly-Arounds)
# ======================================================================
def plan_scp_moving_cone(
    p_start,
    target_traj,    
    p_obs,
    r_obs,
    docking_axis,   
    cone_angle_deg,
    dt
):
    print(f"SCP: Planning Intercept for Axis {docking_axis}...")
    N = len(target_traj)
    
    # Dynamics
    A = np.eye(6); A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
    B = np.zeros((6,3)); B[0,0]=0.5*dt**2; B[1,1]=0.5*dt**2; B[2,2]=0.5*dt**2; B[3,0]=dt; B[4,1]=dt; B[5,2]=dt

    # Cone Geometry
    n_approach = docking_axis / np.linalg.norm(docking_axis)
    cos_theta = np.cos(np.deg2rad(cone_angle_deg))
    
    p_target_final = target_traj[-1, 0:3]
    v_target_final = target_traj[-1, 3:6]

    # --- SMART INITIALIZATION (THE FIX) ---
    # 1. Calculate Entry Point (3.0m back from the final target position)
    # If Axis=[-1,0,0] (Left), n_approach=[-1,0,0]. We want point at +X relative to target.
    # So we subtract n_approach.
    p_entry = p_target_final - n_approach * 3.0
    
    # 2. Detect "Fly-Around" Condition
    # Vector from Start to Goal
    vec_to_goal = p_target_final - p_start
    # If we are flying AGAINST the approach axis (dot product < 0), we are on the wrong side.
    is_fly_around = np.dot(vec_to_goal, n_approach) < 0
    
    x_ref = np.zeros((6, N))
    
    if is_fly_around:
        print(">> Logic: Fly-Around Detected. Adding clearance waypoint.")
        # Waypoint: High above the obstacle/target to avoid collision during U-Turn
        # Midpoint between start and entry, but HIGH up (Z=1.5)
        p_mid = (p_start + p_entry) / 2
        p_mid[2] = 2.0 # Fly over everything
        
        idx_1 = int(0.4 * N) # Reach clearance
        idx_2 = int(0.7 * N) # Reach entry
        
        for k in range(N):
            if k < idx_1: # Start -> Clearance
                alpha = k/idx_1
                x_ref[0:3, k] = (1-alpha)*p_start + alpha*p_mid
            elif k < idx_2: # Clearance -> Entry
                alpha = (k-idx_1)/(idx_2-idx_1)
                x_ref[0:3, k] = (1-alpha)*p_mid + alpha*p_entry
            else: # Entry -> Target
                alpha = (k-idx_2)/(N-idx_2)
                curr_targ = target_traj[k, 0:3]
                x_ref[0:3, k] = (1-alpha)*p_entry + alpha*curr_targ
                
    else:
        print(">> Logic: Direct Approach Detected.")
        idx_entry = int(0.6 * N)
        for k in range(N):
            if k < idx_entry: # Start -> Entry
                alpha = k/idx_entry
                x_ref[0:3, k] = (1-alpha)*p_start + alpha*p_entry
                x_ref[2, k] += 1.0*np.sin(np.pi*alpha) # Simple obstacle hop
            else: # Entry -> Target
                alpha = (k-idx_entry)/(N-idx_entry)
                curr_targ = target_traj[k, 0:3]
                x_ref[0:3, k] = (1-alpha)*p_entry + alpha*curr_targ

    # --- OPTIMIZATION ---
    max_iters = 15
    for iteration in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack_obs = cp.Variable(N, nonneg=True)
        slack_cone = cp.Variable(N, nonneg=True)

        cost = cp.sum_squares(u) + 1000*cp.sum(slack_obs) + 1000*cp.sum(slack_cone)
        constraints = []

        constraints += [x[:,0] == np.hstack([p_start, np.zeros(3)])]
        
        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]

        # Terminal Intercept (Soft Dock)
        # Target a small offset to avoid crashing into Target CoM
        dock_offset = -n_approach * 0.30 
        constraints += [cp.norm(x[0:3,-1] - (p_target_final + dock_offset)) <= 0.05]
        constraints += [cp.norm(x[3:6,-1] - v_target_final) <= 0.05]

        for k in range(1, N):
            # 1. Obstacle
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)
            n_obs = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [n_obs @ (x[0:3,k] - p_obs) >= r_obs - slack_obs[k]]

            # 2. MOVING DOCKING CONE
            # Enforce only in the final approach phase (last 30%)
            if k > 0.7 * N:
                p_rel = x[0:3,k] - target_traj[k, 0:3]
                
                # Axial distance relative to Moving Target
                # If n_approach is the "Axis", we want p_rel to be opposed to it
                dist_axial = -n_approach @ p_rel
                
                constraints += [dist_axial >= 0]
                constraints += [cp.norm(p_rel) * cos_theta <= dist_axial + slack_cone[k]]

        # Relaxed Trust Region for long fly-arounds
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.5]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try: prob.solve(solver=cp.CLARABEL)
        except: 
            try: prob.solve(solver=cp.ECOS)
            except: prob.solve(solver=cp.SCS)

        if x.value is None:
            print(f"Solver Failed iter {iteration}")
            break
            
        diff = np.linalg.norm(x.value - x_ref)
        x_ref = x.value.copy()
        
        if diff < 0.15: # Slightly looser tolerance for complex paths
            print(f"Converged iter {iteration}")
            break

    return x_ref[0:3,:].T

# ======================================================================
# 3. DYNAMIC VISUALIZATION
# ======================================================================
class MovingConeVisualizer:
    def __init__(self, client, axis, angle_deg, length=1.5):
        self.client = client
        self.axis = axis / np.linalg.norm(axis)
        self.length = length
        self.tan_alpha = np.tan(np.deg2rad(angle_deg))
        self.lines = [] 
        
        # Precompute cone rim vectors
        if np.abs(self.axis[2]) < 0.9: ref = np.array([0,0,1])
        else: ref = np.array([0,1,0])
        u = np.cross(self.axis, ref); u = u/np.linalg.norm(u)
        v = np.cross(self.axis, u)
        
        # Cone opens in direction -axis
        self.cone_vectors = []
        center_dist = -self.axis * self.length
        radius = self.length * self.tan_alpha
        
        for phi in np.linspace(0, 2*np.pi, 12):
            rim_point = center_dist + radius * (u*np.cos(phi) + v*np.sin(phi))
            self.cone_vectors.append(rim_point)
            
    def update(self, origin):
        for line_id in self.lines:
            p.removeUserDebugItem(line_id, physicsClientId=self.client)
        self.lines = []
        
        # Draw rim lines
        for vec in self.cone_vectors:
            end_pos = origin + vec
            line_id = p.addUserDebugLine(origin, end_pos, [0,1,0], 2, physicsClientId=self.client)
            self.lines.append(line_id)
            
        # Draw central axis
        axis_end = origin - self.axis * self.length
        line_id = p.addUserDebugLine(origin, axis_end, [0,1,0], 3, physicsClientId=self.client)
        self.lines.append(line_id)

def draw_trajectory(traj, color, client):
    for i in range(len(traj)-1):
        p.addUserDebugLine(traj[i, 0:3], traj[i+1, 0:3], color, 2, physicsClientId=client)

# ======================================================================
# 4. MAIN RUN
# ======================================================================
def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION = 30  # Increased to 30s to allow "Fly Around"

    CHASER_START = np.array([-2.5, 0.0, 0.6])
    P_OBS        = np.array([-0.5, 0.1, 0.6])
    R_OBS        = 0.5
    
    # --- TEST CONFIGURATION ---
    # Try [-1, 0, 0] (Inverted) -> Should fly over and come back
    DOCKING_AXIS = np.array([-1.0, 0.0, 0.0]) 
    CONE_ANGLE   = 20

    # 1. Generate Target Future
    target_full_traj = generate_target_trajectory(DURATION, CTRL_FREQ)
    
    # 2. Plan
    chaser_traj = plan_scp_moving_cone(
        p_start=CHASER_START,
        target_traj=target_full_traj,
        p_obs=P_OBS,
        r_obs=R_OBS,
        docking_axis=DOCKING_AXIS,
        cone_angle_deg=CONE_ANGLE,
        dt=1/CTRL_FREQ
    )

    # 3. Sim Setup
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
    
    # Visuals
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=R_OBS), 
                      p.createVisualShape(p.GEOM_SPHERE, radius=R_OBS, rgbaColor=[1,0,0,0.4]), 
                      P_OBS, physicsClientId=PYB)
    p.setCollisionFilterPair(0, 1, -1, -1, 0, physicsClientId=PYB) # Disable drone collision

    cone_viz = MovingConeVisualizer(PYB, DOCKING_AXIS, CONE_ANGLE)
    
    # Draw Planned Path (Blue)
    draw_trajectory(chaser_traj, [0,0,1], PYB)
    # Draw Target Path (Red)
    draw_trajectory(target_full_traj, [1,0,0], PYB)

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    START = time.time()

    for i in range(len(chaser_traj)):
        obs, _, _, _, _ = env.step(action)
        
        cone_viz.update(obs[1][0:3])

        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], chaser_traj[i], np.zeros(3))
        
        t_pos = target_full_traj[i, 0:3]
        t_vel = target_full_traj[i, 3:6]
        action[1], _, _ = ctrl[1].computeControlFromState(env.CTRL_TIMESTEP, obs[1], t_pos, t_vel)

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
        
    print("Mission Complete.")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    run()