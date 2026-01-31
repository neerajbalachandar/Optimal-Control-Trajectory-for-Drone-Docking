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
# 0. CONFIGURATION & CONSTANTS
# ======================================================================
DOCKING_AXIS = np.array([0.0, 0.0,-1.0]) 
CONE_ANGLE   = 30    # Degrees
DURATION_SEC = 20    # Time for trajectory
SAFETY_R     = 0.1   # Safety Radius (Hull size)
ALPHA_LIMIT  = 0.9  # Collision Trigger Threshold (Distance/Radius ratio)

# Wind Configuration
WIND_NOMINAL  = np.array([0.0, -0.0, -0.0]) # Constant drift
WIND_GUST_AMP = 0.0                         # Random noise magnitude

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. ROBUST DCOL SOLVER (Safety Check)
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    """
    Calculates the collision scaling factor alpha.
    alpha < 1.0 implies collision.
    """
    # 1. Fast Analytical Check
    dist = np.linalg.norm(p1 - p2)
    alpha_analytic = dist / (r1 + r2)
    
    # 2. Optimization Check (Formal method fallback)
    # We use analytical primarily for speed in Python, 
    # but this structure allows swapping in the CVX solver if shapes become complex.
    contact_pt = p1 + (p2 - p1) * (r1 / (r1 + r2))
    return alpha_analytic, contact_pt

# ======================================================================
# 2. SCP TRAJECTORY PLANNER
# ======================================================================
def plan_scp_docking(p_start, v_start, p_goal, p_obs, r_obs, docking_axis, cone_angle_deg, N, dt):
    """
    Generates a trajectory satisfying Dynamics, Obstacles, and Docking Cone.
    Includes 'Smart Entry' initialization to help the solver find the cone entrance.
    """
    print(f"[SCP] Planning N={N} from {p_start} with Vel {v_start}...")
    
    # Dynamics (Double Integrator 3D)
    A = np.eye(6); A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
    B = np.zeros((6,3)); B[0,0]=0.5*dt**2; B[1,1]=0.5*dt**2; B[2,2]=0.5*dt**2; B[3,0]=dt; B[4,1]=dt; B[5,2]=dt

    # Cone Math
    n_approach = docking_axis / np.linalg.norm(docking_axis)
    cos_theta = np.cos(np.deg2rad(cone_angle_deg))

    # --- SMART INITIALIZATION (Hybrid) ---
    # Define an "Entry Point" 2.5m out along the docking axis
    p_entry = p_goal - n_approach * 2.5
    
    x_ref = np.zeros((6, N))
    split_idx = int(0.6 * N) # 60% of time to reach entry point
    
    for k in range(N):
        if k < split_idx:
            # Phase 1: Go to Entry Point (Bezier-ish curve to clear obstacles)
            alpha = k / split_idx
            # Pos
            x_ref[0:3, k] = (1 - alpha) * p_start + alpha * p_entry
            # Add a small Z-hop to clear low obstacles if starting far away
            if np.linalg.norm(p_start - p_entry) > 1.0:
                x_ref[2, k] += 0.5 * np.sin(np.pi * alpha)
            # Vel
            x_ref[3:6, k] = v_start * (1 - alpha) # Decay initial velocity
        else:
            # Phase 2: Straight Approach (Entry -> Goal)
            alpha = (k - split_idx) / (N - split_idx)
            x_ref[0:3, k] = (1 - alpha) * p_entry + alpha * p_goal

    # --- SCP LOOP ---
    max_iters = 15
    solved_traj = None

    for iteration in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack_obs = cp.Variable(N, nonneg=True)
        slack_cone = cp.Variable(N, nonneg=True)

        # Cost: Minimal Control + Heavy penalties on Slacks
        cost = 0.1*cp.sum_squares(u) + 10000*cp.sum(slack_obs) + 10000*cp.sum(slack_cone)
        constraints = []

        # 1. Initial State
        constraints += [x[:,0] == np.hstack([p_start, v_start])]
        
        # 2. Dynamics
        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]
            constraints += [cp.norm(u[:,k], 2) <= 12.0] # Actuation Limit

        # 3. Terminal Constraints
        constraints += [cp.norm(x[0:3,-1] - p_goal) <= 0.05] # Position Accuracy
        constraints += [cp.norm(x[3:6,-1]) <= 0.1]           # Velocity near zero

        # 4. State Constraints (Linearized)
        for k in range(1, N):
            # A. Obstacle Avoidance
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)
            n_obs_vec = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [n_obs_vec @ (x[0:3,k] - p_obs) >= r_obs - slack_obs[k]]

            # B. Docking Cone (Active only in approach phase)
            # Logic: Project p_rel onto -n_approach (distance 'deep' into cone)
            # Radius at that depth = dist_long * tan(theta).
            # Constraint: ||p_lateral|| <= radius
            # Simplified SOCP form: ||p_rel|| * cos(theta) <= dist_long
            if k > split_idx:
                p_rel = x[0:3,k] - p_goal 
                dist_long = -n_approach @ p_rel
                
                constraints += [dist_long >= 0]
                constraints += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]

        # 5. Trust Region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            try: prob.solve(solver=cp.ECOS)
            except: prob.solve(solver=cp.SCS)

        if x.value is None:
            print(f"  > Iter {iteration}: Solver Failed.")
            break
            
        diff = np.linalg.norm(x.value - x_ref)
        x_ref = x.value.copy()
        
        if diff < 0.1:
            print(f"  > Iter {iteration}: Converged.")
            solved_traj = x_ref[0:3,:].T
            break

    return solved_traj

# ======================================================================
# 3. VISUALIZATION HELPERS
# ======================================================================
def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)

def draw_planned_path(traj, client, target_pos, axis, angle):
    """
    Clears old debug lines, RESTORES the cone, and draws the new path.
    """
    # 1. Clear everything (Old path, old cone, old collision lines)
    p.removeAllUserDebugItems(physicsClientId=client)
    
    # 2. Restore the Cone immediately
    visualize_docking_cone(target_pos, axis, angle, client=client)
    
    # 3. Draw the new Trajectory
    if traj is None: return
    for i in range(len(traj) - 1):
        p.addUserDebugLine(traj[i], traj[i+1], [0, 0, 1], 3, physicsClientId=client)

def create_spherical_obstacle(p_obs, r_obs, client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_obs, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r_obs, rgbaColor=[1,0,0,0.8], physicsClientId=client)
    p.createMultiBody(0, col, vis, p_obs, physicsClientId=client)
    
def visualize_docking_cone(p_goal, axis, angle_deg, length=2.0, client=0):
    axis = axis / np.linalg.norm(axis)
    cone_dir_main = -axis
    theta = np.deg2rad(angle_deg)

    # Construct Basis (unchanged)
    if np.abs(axis[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([0, 1, 0])
    u = np.cross(axis, ref); u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    
    # Draw Rim
    for phi in np.linspace(0, 2*np.pi, 12):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir_main * np.cos(theta) + radial * np.sin(theta)
        end_pos = p_goal + vec * length
        p.addUserDebugLine(p_goal, end_pos, [0,1,0], 2, physicsClientId=client)
    
    # Draw Center Axis
    p.addUserDebugLine(p_goal, p_goal + cone_dir_main*length, [0,1,0], 4, physicsClientId=client)

# ======================================================================
# 4. MAIN INTEGRATED EXECUTION
# ======================================================================
def run():
    # --- SIMULATION PARAMETERS ---
    SIM_FREQ = 240
    CTRL_FREQ = 48
    
    # --- SCENARIO SETUP ---
    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 1.5, 0.0, 0.6]) 
    P_OBS        = np.array([-0.5, 0.1, 0.6]) 
    R_OBS        = 0.6

    # 1. INITIAL PLAN
    print("\n[INIT] Calculating Initial Trajectory...")
    current_plan = plan_scp_docking(
        p_start=CHASER_START,
        v_start=np.zeros(3),
        p_goal=TARGET_POS,
        p_obs=P_OBS,
        r_obs=R_OBS + 0.2,
        # r_obs=R_OBS ,
        
        docking_axis=DOCKING_AXIS,
        cone_angle_deg=CONE_ANGLE,
        N=int(CTRL_FREQ * DURATION_SEC),
        dt=1/CTRL_FREQ
    )

    # 2. ENVIRONMENT SETUP (With Downwash)
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, TARGET_POS]),
        initial_rpys=np.zeros((2,3)),
        physics=Physics.PYB_DW,   # <--- DOWNWASH ENABLED
        neighbourhood_radius=0.3,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False
    )
    
    

       
       
    PYB_CLIENT = env.getPyBulletClient()

    # 3. VISUALS SETUP
    create_spherical_obstacle(P_OBS, R_OBS, PYB_CLIENT)
    visualize_docking_cone(TARGET_POS, DOCKING_AXIS, CONE_ANGLE, client=PYB_CLIENT)
    
    # Safety Hulls (Visual only, physics handled manually)
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB_CLIENT) 
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB_CLIENT) 

    if current_plan is not None:
        draw_planned_path(current_plan, PYB_CLIENT, TARGET_POS, DOCKING_AXIS, CONE_ANGLE)
    else:
        print("[ERROR] Initial Plan Failed!")
        env.close(); return

    # 4. CONTROLLERS
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    # 5. STATE VARIABLES
    state = STATE_TRACKING
    traj_idx = 0
    START = time.time()
    
    # Retreat Logic Variables
    backoff_start_pos = None
    backoff_end_pos   = None
    backoff_t_start   = 0
    BACKOFF_DURATION  = 2.5 

    # --- MAIN LOOP ---
    while True:
        # A. STEP SIMULATION
        obs, _, _, _, _ = env.step(action)
        
        # B. EXTRACT STATE
        p_chaser = obs[0][0:3]
        v_chaser = obs[0][10:13]
        p_target = obs[1][0:3]

        # Update Visual Hulls
        update_hull(hull_c, p_chaser, PYB_CLIENT)
        update_hull(hull_t, p_target, PYB_CLIENT)

        # C. APPLY ENVIRONMENTAL DISTURBANCE (WIND)
        gust = np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
        current_wind_force = WIND_NOMINAL + gust
        
        # Apply to Chaser (ID 0)
        p.applyExternalForce(
            objectUniqueId=env.DRONE_IDS[0],
            linkIndex=-1,
            forceObj=current_wind_force,
            posObj=p_chaser,
            flags=p.WORLD_FRAME,
            physicsClientId=PYB_CLIENT
        )
        
        # Visualize Wind (Yellow line)
        p.addUserDebugLine(p_chaser, p_chaser + current_wind_force*0.5, [1, 1, 0], 2, 0.1, physicsClientId=PYB_CLIENT)

        # D. FINITE STATE MACHINE (FSM)
        if state == STATE_TRACKING:
            # 1. SAFETY CHECK (DCOL)
            # Only check if we are reasonably close to avoid false positives at start
            alpha, contact_pt = solve_dcol_scaling(p_chaser, SAFETY_R, p_target, SAFETY_R)
            
            # TRIGGER CONDITION: Alpha < Limit AND we aren't already docked (end of traj)
            remaining_steps = len(current_plan) - traj_idx
            if alpha < ALPHA_LIMIT and remaining_steps > 20:
                print(f"\n[SAFETY TRIGGER] Alpha {alpha:.3f} < {ALPHA_LIMIT}. WIND CAUSED DEVIATION! BACKING OFF.")
                
                # Visual Alert
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB_CLIENT)
                p.addUserDebugLine(p_chaser, contact_pt, [1,0,0], 3, physicsClientId=PYB_CLIENT)
                
                # Setup Smooth Retreat (Vector away from target + Up)
                backoff_start_pos = p_chaser.copy()
                vec_away = p_chaser - p_target
                if np.linalg.norm(vec_away) < 0.01: vec_away = np.array([-1.0, 0, 0])
                vec_away = vec_away / np.linalg.norm(vec_away)
                
                backoff_end_pos = p_target + vec_away * (SAFETY_R * 3.0) + np.array([0,0,0.5])
                backoff_t_start = time.time()
                
                state = STATE_BACKING_OFF

            # 2. TRACKING
            elif current_plan is not None and traj_idx < len(current_plan):
                target_pt = current_plan[traj_idx]
                target_vel = (current_plan[traj_idx] - current_plan[traj_idx-1])*CTRL_FREQ if traj_idx > 0 else np.zeros(3)
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], target_pt, target_vel)
                traj_idx += 1
            else:
                # Hover at Goal
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], TARGET_POS, np.zeros(3))

        elif state == STATE_BACKING_OFF:
            # Smooth Interpolation to safety point
            t_elapsed = time.time() - backoff_t_start
            progress = min(t_elapsed / BACKOFF_DURATION, 1.0)
            alpha_t = progress * progress * (3 - 2 * progress) # Smoothstep
            
            current_setpoint = (1 - alpha_t) * backoff_start_pos + alpha_t * backoff_end_pos
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], current_setpoint, np.zeros(3))
            
            if progress >= 1.0:
                print(">>> Safety Reached. Replanning...")
                state = STATE_REPLANNING

        elif state == STATE_REPLANNING:
            # Hover while computing
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser, np.zeros(3))
            
            # Re-run SCP from CURRENT position/velocity
            new_plan = plan_scp_docking(
                p_start=p_chaser, 
                v_start=v_chaser, 
                p_goal=TARGET_POS, 
                p_obs=P_OBS, r_obs=R_OBS+ 0.2,
                # p_obs=P_OBS, r_obs=R_OBS,
                
                docking_axis=DOCKING_AXIS, cone_angle_deg=CONE_ANGLE,
                N=int(CTRL_FREQ * 15), dt=1/CTRL_FREQ # Shorter horizon for re-approach
            )
            
            if new_plan is not None:
                current_plan = new_plan
                traj_idx = 0
    
                draw_planned_path(current_plan, PYB_CLIENT, TARGET_POS, DOCKING_AXIS, CONE_ANGLE)
                p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB_CLIENT)
                print(">>> Replan Successful. Resuming Approach.")
                state = STATE_TRACKING
            else:
                print(">>> Replan Failed. Trying again...")

        # Target (Static Hover)
        action[1], _, _ = ctrl[1].computeControlFromState(env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3))

        env.render()
        sync(traj_idx, START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()