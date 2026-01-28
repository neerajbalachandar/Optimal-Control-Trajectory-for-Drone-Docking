import time
import numpy as np
import cvxpy as cp
import pybullet as p
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

# ======================================================================
# 0. CONFIGURATION & CONSTANTS
# ======================================================================
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    # Degrees
DURATION_SEC = 20    # Time for trajectory
SAFETY_R     = 0.1   # Safety Radius (Hull size)
ALPHA_LIMIT  = 1.05  # Collision Trigger Threshold

# Physics Constants (Standard CF2X)
MASS = 0.027
G = 9.8
KF = 3.16e-10 # Motor Force Constant
KM = 7.94e-12 # Motor Moment Constant
ARM_LENGTH = 0.0397

# Wind Configuration
WIND_NOMINAL  = np.array([0.3, -0.1, 0.0]) 
WIND_GUST_AMP = 0.05                          

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. ROBUST DCOL SOLVER (Safety Check)
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    alpha_analytic = dist / (r1 + r2)
    contact_pt = p1 + (p2 - p1) * (r1 / (r1 + r2))
    return alpha_analytic, contact_pt

# ======================================================================
# 2. SCP TRAJECTORY PLANNER (Returning X and U)
# ======================================================================
def plan_scp_docking(p_start, v_start, p_goal, p_obs, r_obs, docking_axis, cone_angle_deg, N, dt):
    """
    Returns:
       x_opt: (6, N) State trajectory
       u_opt: (3, N-1) Control input (Accelerations)
    """
    print(f"[SCP] Planning N={N}...")
    
    # Dynamics (Double Integrator 3D)
    A = np.eye(6); A[0,3]=dt; A[1,4]=dt; A[2,5]=dt
    B = np.zeros((6,3)); B[0,0]=0.5*dt**2; B[1,1]=0.5*dt**2; B[2,2]=0.5*dt**2; B[3,0]=dt; B[4,1]=dt; B[5,2]=dt

    # Cone Math
    n_approach = docking_axis / np.linalg.norm(docking_axis)
    cos_theta = np.cos(np.deg2rad(cone_angle_deg))

    # --- SMART INITIALIZATION ---
    p_entry = p_goal - n_approach * 2.5
    x_ref = np.zeros((6, N))
    split_idx = int(0.6 * N) 
    
    for k in range(N):
        if k < split_idx:
            alpha = k / split_idx
            x_ref[0:3, k] = (1 - alpha) * p_start + alpha * p_entry
            if np.linalg.norm(p_start - p_entry) > 1.0:
                x_ref[2, k] += 1.0 * np.sin(np.pi * alpha)
            x_ref[3:6, k] = v_start * (1 - alpha)
        else:
            alpha = (k - split_idx) / (N - split_idx)
            x_ref[0:3, k] = (1 - alpha) * p_entry + alpha * p_goal

    # --- SCP LOOP ---
    max_iters = 15
    x_opt, u_opt = None, None

    for iteration in range(max_iters):
        x = cp.Variable((6,N))
        u = cp.Variable((3,N-1))
        slack_obs = cp.Variable(N, nonneg=True)
        slack_cone = cp.Variable(N, nonneg=True)

        cost = 0.1*cp.sum_squares(u) + 10000*cp.sum(slack_obs) + 10000*cp.sum(slack_cone)
        constraints = []

        # 1. Initial State
        constraints += [x[:,0] == np.hstack([p_start, v_start])]
        
        # 2. Dynamics
        for k in range(N-1):
            constraints += [x[:,k+1] == A@x[:,k] + B@u[:,k]]
            constraints += [cp.norm(u[:,k], 2) <= 15.0] # Slightly higher actuation limit

        # 3. Terminal Constraints
        constraints += [cp.norm(x[0:3,-1] - p_goal) <= 0.05]
        constraints += [cp.norm(x[3:6,-1]) <= 0.1]

        # 4. State Constraints (Linearized)
        for k in range(1, N):
            # A. Obstacle
            p_ref = x_ref[0:3,k]
            vec = p_ref - p_obs
            dist = np.linalg.norm(vec)
            n_obs_vec = vec/dist if dist > 1e-3 else np.array([0,1,0])
            constraints += [n_obs_vec @ (x[0:3,k] - p_obs) >= r_obs - slack_obs[k]]

            # B. Cone
            if k > split_idx:
                p_rel = x[0:3,k] - p_goal 
                dist_long = -n_approach @ p_rel
                constraints += [dist_long >= 0]
                constraints += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]

        # 5. Trust Region
        for k in range(N):
            constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try: prob.solve(solver=cp.CLARABEL)
        except: prob.solve(solver=cp.SCS)

        if x.value is None: break
            
        diff = np.linalg.norm(x.value - x_ref)
        x_ref = x.value.copy()
        
        if diff < 0.1:
            print(f"  > Iter {iteration}: Converged.")
            x_opt = x.value
            u_opt = u.value
            break

    return x_opt, u_opt

# ======================================================================
# 3. CONTROL CONVERSION (ACCEL -> RPM) 
# ======================================================================
def get_rpm_from_vector(acc_des, current_quat, current_ang_vel):
    """
    Converts SCP Acceleration Vector (u) into 4 Motor RPMs.
    Now includes Inertia Scaling to prevent torque saturation.
    """
    # --- CONSTANTS (Standard CF2X) ---
    KF = 3.16e-10            # Force constant (N / RPM^2)
    KM = 7.94e-12            # Moment constant (Nm / RPM^2)
    L  = 0.0397              # Arm length (m)
    
    # Inertia Matrix (Diagonal) for Crazyflie 2.x
    I = np.diag([1.4e-5, 1.4e-5, 2.17e-5]) 

    # 1. Target Force (Newtons)
    g_vec = np.array([0, 0, 9.8])
    target_force_vec = MASS * (acc_des + g_vec)
    
    # Project onto body Z-axis
    r_curr = R.from_quat(current_quat).as_matrix()
    z_body = r_curr[:, 2]
    thrust_newtons = np.dot(target_force_vec, z_body)
    thrust_newtons = np.maximum(thrust_newtons, 0.0) 

    # 2. Orientation Control (Geometric Controller)
    z_des = target_force_vec / (np.linalg.norm(target_force_vec) + 1e-6)
    
    # Heading constraint
    x_curr = r_curr[:, 0]
    y_des = np.cross(z_des, x_curr)
    y_des /= (np.linalg.norm(y_des) + 1e-6)
    x_des = np.cross(y_des, z_des)
    R_des = np.stack([x_des, y_des, z_des], axis=1)
    
    # Rotation Error
    e_R_mat = 0.5 * (R_des.T @ r_curr - r_curr.T @ R_des)
    e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
    
    # --- GAIN CORRECTION IS HERE ---
    # We use normalized gains (approx 2000-3000 is good for normalized),
    # BUT we must multiply by Inertia to get Torques (Nm).
    k_R_norm = np.array([3000, 3000, 3000]) # Normalized Stiffness
    k_w_norm = np.array([200, 200, 200])    # Normalized Damping

    # Compute Desired Angular Acceleration (normalized)
    # Then multiply by Inertia to get Moments (Nm)
    # M = I * (-k_R * e_R - k_w * w)
    
    desired_ang_acc = -k_R_norm * e_R - k_w_norm * current_ang_vel
    moments = I @ desired_ang_acc  # <--- SCALING APPLIED

    # 3. MIXER (Force/Torque -> RPM)
    # T_term = Force / 4*KF
    T_term = thrust_newtons / (4 * KF)
    
    # Torque terms = Moment / Arm / KF (or KM for yaw)
    # Note: The constants below handle the geometry
    R_term = moments[0] / (2 * np.sqrt(2) * L * KF)
    P_term = moments[1] / (2 * np.sqrt(2) * L * KF)
    Y_term = moments[2] / (4 * KM)

    w_sq = np.zeros(4)
    w_sq[0] = T_term - R_term - P_term - Y_term # FR
    w_sq[1] = T_term + R_term + P_term - Y_term # RL
    w_sq[2] = T_term - R_term + P_term + Y_term # FL
    w_sq[3] = T_term + R_term - P_term + Y_term # RR

    w_sq = np.clip(w_sq, 0, 22000**2) 
    return np.sqrt(w_sq)
# ======================================================================
# 4. VISUALIZATION HELPERS
# ======================================================================
def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)

def draw_planned_path(traj, client, target_pos, axis, angle):
    p.removeAllUserDebugItems(physicsClientId=client)
    visualize_docking_cone(target_pos, axis, angle, client=client)
    if traj is None: return
    for i in range(traj.shape[1] - 1):
        p.addUserDebugLine(traj[0:3,i], traj[0:3,i+1], [0, 0, 1], 3, physicsClientId=client)

def create_spherical_obstacle(p_obs, r_obs, client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_obs, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r_obs, rgbaColor=[1,0,0,0.4], physicsClientId=client)
    p.createMultiBody(0, col, vis, p_obs, physicsClientId=client)

def visualize_docking_cone(p_goal, axis, angle_deg, length=2.0, client=0):
    axis = axis / np.linalg.norm(axis)
    cone_dir_main = -axis 
    theta = np.deg2rad(angle_deg)
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    for phi in np.linspace(0, 2*np.pi, 12):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir_main * np.cos(theta) + radial * np.sin(theta)
        end_pos = p_goal + vec * length
        p.addUserDebugLine(p_goal, end_pos, [0,1,0], 2, physicsClientId=client)
    p.addUserDebugLine(p_goal, p_goal + cone_dir_main*length, [0,1,0], 4, physicsClientId=client)

# ======================================================================
# 5. MAIN INTEGRATED EXECUTION
# ======================================================================
def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    
    CHASER_START = np.array([-2.5, 0.0, 0.6])
    TARGET_POS   = np.array([ 1.5, 0.0, 0.6]) 
    P_OBS        = np.array([-0.5, 0.1, 0.6]) 
    R_OBS        = 0.6

    # 1. INITIAL PLAN
    print("\n[INIT] Calculating Initial Trajectory...")
    x_plan, u_plan = plan_scp_docking(
        p_start=CHASER_START, v_start=np.zeros(3),
        p_goal=TARGET_POS, p_obs=P_OBS, r_obs=R_OBS + 0.2,
        docking_axis=DOCKING_AXIS, cone_angle_deg=CONE_ANGLE,
        N=int(CTRL_FREQ * DURATION_SEC), dt=1/CTRL_FREQ
    )

    # 2. ENVIRONMENT
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, TARGET_POS]),
        initial_rpys=np.zeros((2,3)),
        physics=Physics.PYB_DW,   
        neighbourhood_radius=0.2,
        pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=True, obstacles=False
    )
    PYB_CLIENT = env.getPyBulletClient()

    # 3. VISUALS
    create_spherical_obstacle(P_OBS, R_OBS, PYB_CLIENT)
    visualize_docking_cone(TARGET_POS, DOCKING_AXIS, CONE_ANGLE, client=PYB_CLIENT)
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB_CLIENT) 
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB_CLIENT) 

    if x_plan is not None:
        draw_planned_path(x_plan, PYB_CLIENT, TARGET_POS, DOCKING_AXIS, CONE_ANGLE)
    else:
        print("[ERROR] Initial Plan Failed!"); env.close(); return

    # 4. VARIABLES
    action = np.zeros((2,4))
    state = STATE_TRACKING
    traj_idx = 0
    START = time.time()
    
    backoff_start_pos = None
    backoff_end_pos   = None
    backoff_t_start   = 0

    # --- MAIN LOOP ---
    while True:
        # A. STEP
        obs, _, _, _, _ = env.step(action)
        
        # B. EXTRACT STATE
        p_chaser = obs[0][0:3]
        v_chaser = obs[0][10:13]
        q_chaser = obs[0][3:7]   # Quaternion
        w_chaser = obs[0][13:16] # Angular Vel
        p_target = obs[1][0:3]

        update_hull(hull_c, p_chaser, PYB_CLIENT)
        update_hull(hull_t, p_target, PYB_CLIENT)

        # C. WIND (Disturbance)
        gust = np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
        current_wind_force = WIND_NOMINAL + gust
        p.applyExternalForce(env.DRONE_IDS[0], -1, current_wind_force, p_chaser, p.WORLD_FRAME, PYB_CLIENT)
        p.addUserDebugLine(p_chaser, p_chaser + current_wind_force*0.5, [1, 1, 0], 2, 0.1, physicsClientId=PYB_CLIENT)

        # D. FSM
        if state == STATE_TRACKING:
            # SAFETY CHECK
            alpha, contact_pt = solve_dcol_scaling(p_chaser, SAFETY_R, p_target, SAFETY_R)
            remaining_steps = u_plan.shape[1] - traj_idx if u_plan is not None else 0
            
            if alpha < ALPHA_LIMIT and remaining_steps > 20:
                print(f"\n[SAFETY TRIGGER] Deviation detected. Backing off.")
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB_CLIENT)
                p.addUserDebugLine(p_chaser, contact_pt, [1,0,0], 3, physicsClientId=PYB_CLIENT)
                
                backoff_start_pos = p_chaser.copy()
                vec_away = p_chaser - p_target
                if np.linalg.norm(vec_away) < 0.01: vec_away = np.array([-1.0, 0, 0])
                vec_away = vec_away / np.linalg.norm(vec_away)
                backoff_end_pos = p_target + vec_away * (SAFETY_R * 3.0) + np.array([0,0,0.5])
                backoff_t_start = time.time()
                state = STATE_BACKING_OFF

            # CONTROL APPLICATION (Using SCP 'u')
            elif u_plan is not None and traj_idx < u_plan.shape[1]:
                # 1. Get Acceleration from SCP (Feedforward)
                acc_des = u_plan[:, traj_idx]
                
                # 2. Add minimal feedback for position drift (Proportional P only)
                # Since we removed the PID, if we don't have this small correction, 
                # the wind will drift us away infinitely because SCP 'u' is open-loop.
                pos_error = x_plan[0:3, traj_idx] - p_chaser
                vel_error = x_plan[3:6, traj_idx] - v_chaser
                acc_correction = 2.0 * pos_error + 1.5 * vel_error
                
                # 3. Convert Total Acceleration to RPM
                action[0] = get_rpm_from_vector(acc_des + acc_correction, q_chaser, w_chaser)
                
                traj_idx += 1
            else:
                # Hover at Goal
                # Simple P-loop for hover since SCP is done
                err = TARGET_POS - p_chaser
                acc_hover = 1.0 * err - 0.5 * v_chaser
                action[0] = get_rpm_from_vector(acc_hover, q_chaser, w_chaser)

        elif state == STATE_BACKING_OFF:
            # Simple manual trajectory
            t_elapsed = time.time() - backoff_t_start
            progress = min(t_elapsed / 2.5, 1.0)
            alpha_t = progress * progress * (3 - 2 * progress)
            
            curr_setpoint = (1 - alpha_t) * backoff_start_pos + alpha_t * backoff_end_pos
            
            # Simple P-control for retreat
            err = curr_setpoint - p_chaser
            acc_retreat = 2.0 * err - 1.0 * v_chaser
            action[0] = get_rpm_from_vector(acc_retreat, q_chaser, w_chaser)
            
            if progress >= 1.0:
                print(">>> Safety Reached. Replanning...")
                state = STATE_REPLANNING

        elif state == STATE_REPLANNING:
            # Hover
            action[0] = get_rpm_from_vector(-1.0*v_chaser, q_chaser, w_chaser)
            
            x_plan, u_plan = plan_scp_docking(
                p_start=p_chaser, v_start=v_chaser,
                p_goal=TARGET_POS, p_obs=P_OBS, r_obs=R_OBS+ 0.2,
                docking_axis=DOCKING_AXIS, cone_angle_deg=CONE_ANGLE,
                N=int(CTRL_FREQ * 15), dt=1/CTRL_FREQ
            )
            
            if x_plan is not None:
                traj_idx = 0
                draw_planned_path(x_plan, PYB_CLIENT, TARGET_POS, DOCKING_AXIS, CONE_ANGLE)
                p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB_CLIENT)
                print(">>> Replan Successful. Resuming.")
                state = STATE_TRACKING

        # Target Drone Hover (Using simple helper to keep it steady)
        acc_target = 1.0*(TARGET_POS - p_target) - 0.5*obs[1][10:13]
        action[1] = get_rpm_from_vector(acc_target, obs[1][3:7], obs[1][13:16])

        env.render()
        sync(traj_idx, START, env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()