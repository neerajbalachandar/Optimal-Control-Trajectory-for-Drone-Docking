import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================================
# 0. CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     # Fast planning (5Hz)
HORIZON =20            # 2.0s Horizon
DT_PLAN = 0.1           
DURATION_SEC = 20.0     

# --- DOCKING & SAFETY ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  

# Wind
WIND_NOMINAL  = np.array([0.5, -0.3, -0.1]) 
WIND_GUST_AMP = 0.1                         

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. LINEAR TARGET (Ground Truth)
# ======================================================================
class LinearTarget:
    def get_state(self, t):
        s = np.zeros(6)
        # Moving diagonally up and forward
        s[0] = 0.5 + 0.15 * t  # X
        s[1] = 0.0            # Y
        s[2] = 1.0            # Z
        
        # Velocity
        s[3] = 0.2
        s[4] = 0.0
        s[5] = 0.0
        return s

# ======================================================================
# 2. TARGET EKF
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6)*0.01; self.R = np.eye(3)*0.05
        self.P = np.eye(6)*0.1

    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        for i in range(steps): tmp = Fp @ tmp; fut[i,:] = tmp
        return fut

# ======================================================================
# 3. SAFETY & VISUALS
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    return dist / (r1 + r2)

def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)
    
    
def draw_dynamic_cone(center, axis, angle_deg, client):
    # Initialize static storage for line IDs if it doesn't exist
    if not hasattr(draw_dynamic_cone, "ids"):
        draw_dynamic_cone.ids = []

    # Axis [0,0,-1] -> Approach [0,0,1]
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis 
    
    length = 1.0 # Short cone to visualize immediate docking area
    theta = np.deg2rad(angle_deg)
    
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    line_idx = 0
    # Draw Radial Lines (30 lines)
    for phi in np.linspace(0, 2*np.pi, 30):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        
        if len(draw_dynamic_cone.ids) <= line_idx:
            # Create new line (lifeTime=0 makes it persistent)
            uid = p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, physicsClientId=client)
            draw_dynamic_cone.ids.append(uid)
        else:
            # Update existing line
            p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
        
        line_idx += 1

    # Draw Center Axis Line
    if len(draw_dynamic_cone.ids) <= line_idx:
        uid = p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, physicsClientId=client)
        draw_dynamic_cone.ids.append(uid)
    else:
        p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
    

    
     

# ======================================================================
# 4. ASYNC PLANNER
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt

    def request(self, chaser, preds):
        with self.lock: self.req = (chaser, preds)
    def get(self):
        with self.lock: return self.res

    def run(self):
        print("[Planner] Thread Started.")
        while True:
            data = None
            with self.lock:
                if self.req: data = self.req; self.req = None
            if data:
                try:
                    traj = self._solve_scp(data[0], data[1])
                    with self.lock: self.res = traj
                except: pass
            time.sleep(0.01)

    def _solve_scp(self, start, preds):
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*start[0:3] + al*preds[-1,0:3]
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) # Arc for start
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = preds[-1]

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack_obs = cp.Variable(self.N, nonneg=True)
            slack_cone = cp.Variable(self.N, nonneg=True)
            
            # AIMING FOR EXACT DOCKING (0.0 offset)
            # We want to hit the target itself eventually
            dock_offset = np.zeros(3) 
            
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # --- COST TUNING FOR DOCKING ---
                # 1. Strict Position Tracking
                target_k = preds[k+1, 0:3] + dock_offset
                cost += 50 * cp.sum_squares(x[0:3, k+1] - target_k)
                
                # 2. Minimal Control (but not too cheap, we need agility)
                cost += 0.1 * cp.sum_squares(u[:,k])

            for k in range(1, self.N):
                # Obstacle
                vec = x_ref[0:3,k] - self.p_obs
                n = vec / (np.linalg.norm(vec)+1e-4)
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack_obs[k]]
                
                # Cone Constraint
                p_rel = x[0:3,k] - preds[k, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]
                
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += cp.sum(slack_obs)*1e6 + cp.sum(slack_cone)*1e5
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref

# def plot_performance(history):
#     t = np.array(history['t'])
#     p_c = np.array(history['p_c'])
#     p_t = np.array(history['p_t'])
#     v_c = np.array(history['v_c'])
#     act = np.array(history['action']) # Raw RPMs
    
#     # 1. Error Vector
#     err_vec = p_c - p_t
    
#     # 2. Net Thrust (Sum of RPMs)
#     net_thrust = np.sum(act, axis=1)

#     # --- PLOT 1: 3D TRAJECTORY ---
#     fig1 = plt.figure(figsize=(10, 8))
#     ax1 = fig1.add_subplot(111, projection='3d')
#     ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
#     ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
#     ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
#     ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
#     # ax1.set_title("3D Trajectory")
#     ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
#     ax1.legend()
#     fig1.canvas.manager.set_window_title("Figure 1: 3D Trajectory")

#     # --- PLOT 2: POSITION EVOLUTION ---
#     fig2 = plt.figure(figsize=(10, 6))
#     plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,0], 'b-', label='X')
#     plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,1], 'r-', label='Y')
#     plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,2], 'k-', label='Z')
#     # plt.title("Position Evolution (Target=Dashed)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Position (m)")
#     plt.grid(True)
#     plt.legend()
#     fig2.canvas.manager.set_window_title("Figure 2: Position")

#     # --- PLOT 3: TRACKING ERROR ---
#     fig3 = plt.figure(figsize=(10, 6))
#     plt.plot(t, err_vec[:,0], 'b-', label='X Err')
#     plt.plot(t, err_vec[:,1], 'r-', label='Y Err')
#     plt.plot(t, err_vec[:,2], 'k-', label='Z Err')
#     plt.axhline(0, color='k', linestyle=':', alpha=0.5)
#     # plt.title("Tracking Error (Chaser - Target)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Error (m)")
#     plt.grid(True)
#     plt.legend()
#     fig3.canvas.manager.set_window_title("Figure 3: Tracking Error")

#     # --- PLOT 4: VELOCITY ---
#     fig4 = plt.figure(figsize=(10, 6))
#     plt.plot(t, v_c[:,0], label='Vx')
#     plt.plot(t, v_c[:,1], label='Vy')
#     plt.plot(t, v_c[:,2], label='Vz')
#     # plt.title("Chaser Velocities")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Velocity (m/s)")
#     plt.grid(True)
#     plt.legend()
#     fig4.canvas.manager.set_window_title("Figure 4: Velocity")

#     # --- PLOT 5: NET CONTROL INPUT (THRUST) ---
#     fig5 = plt.figure(figsize=(3.4, 2.4))  # Single-column size

#     start_idx = min(len(t)-1, 50)
#     hover_rpm = np.mean(act[start_idx:], axis=0)
#     act_dev = act - hover_rpm

#     control_norm = np.linalg.norm(act_dev, axis=1)

#     plt.plot(t, control_norm, 'k-', linewidth=1.8)

#     plt.xlabel("Time (s)")
#     plt.ylabel(r"$||\Delta u||$ (RPM)")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(ncol=2)

#     fig5.canvas.manager.set_window_title("Figure 5: Control Effort")
    
    
#     plt.show()
    
    
    

# def plot_performance(history, docking_threshold=0.1):

#     t = np.array(history['t'])
#     p_c = np.array(history['p_c'])
#     p_t = np.array(history['p_t'])
#     v_c = np.array(history['v_c'])
#     act = np.array(history['action'])

#     # -------------------------------------------------
#     # Distance & Error
#     # -------------------------------------------------
#     err_vec = p_c - p_t
#     dist = np.linalg.norm(err_vec, axis=1)

#     # -------------------------------------------------
#     # Control Effort (Non-Dimensionalized)
#     # -------------------------------------------------
#     start_idx = min(len(t)-1, 50)
#     hover_rpm = np.mean(act[start_idx:], axis=0)
    
#     # Raw deviation
#     act_dev = act - hover_rpm
#     control_norm = np.linalg.norm(act_dev, axis=1)
    
#     # Normalize by the magnitude of the hover state to make it dimensionless
#     hover_norm = np.linalg.norm(hover_rpm) + 1e-6 # Add epsilon to avoid div by zero
#     normalized_control = control_norm / hover_norm

#     # -------------------------------------------------
#     # Compute Quantitative Metrics
#     # -------------------------------------------------
#     d0 = dist[0]
#     df = dist[-1]
#     emax = np.max(dist)
#     u_max_nd = np.max(normalized_control) # Now dimensionless

#     idx = np.where(dist < docking_threshold)[0]
#     td = t[idx[0]] if len(idx) > 0 else None

#     print("\n--- PERFORMANCE METRICS ---")
#     print(f"Initial Distance d0        : {d0:.4f} m")
#     print(f"Final Distance df          : {df:.4f} m")
#     print(f"Maximum Tracking Error     : {emax:.4f} m")
#     print(f"Max Control Effort (Norm)  : {u_max_nd:.4f}") # Updated label and format
#     print(f"Docking Time td            : {td:.4f} s" if td else "Docking Time td: Not reached")
#     print("--------------------------------\n")

#     # -------------------------------------------------
#     # --- PLOTS ---
#     # -------------------------------------------------

#     # 1. 3D TRAJECTORY
#     fig1 = plt.figure(figsize=(10, 8))
#     ax1 = fig1.add_subplot(111, projection='3d')
#     ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
#     ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
#     ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
#     ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
#     ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
#     ax1.legend()

#     # 2. POSITION EVOLUTION
#     fig2 = plt.figure(figsize=(10, 6))
#     plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,0], 'b-', label='X')
#     plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,1], 'r-', label='Y')
#     plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
#     plt.plot(t, p_c[:,2], 'k-', label='Z')
#     plt.xlabel("Time (s)")
#     plt.ylabel("Position (m)")
#     plt.grid(True)
#     plt.legend()

#     # 3. TRACKING ERROR (Scaled for Paper)
#     fig3 = plt.figure(figsize=(4.5, 3.4), dpi=300)

#     plt.plot(t, err_vec[:,0], color='#1f77b4', linewidth=2.0, label='Err X')
#     plt.plot(t, err_vec[:,1], color='#d62728', linewidth=2.0, label='Err Y')
#     plt.plot(t, err_vec[:,2], color='black', linewidth=2.0, label='Err Z')

#     plt.axhline(0, color='gray', linestyle=':', linewidth=1)

#     plt.xlabel("Time (s)", fontsize=13)
#     plt.ylabel("Tracking Error (m)", fontsize=13)

#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)

#     plt.grid(True, linestyle=':', alpha=0.4)
#     plt.legend(fontsize=10, frameon=True)

#     plt.tight_layout(pad=1.2)
#     plt.subplots_adjust(bottom=0.18) 

#     # 4. VELOCITY
#     fig4 = plt.figure(figsize=(10, 6))
#     plt.plot(t, v_c[:,0], label='Vx')
#     plt.plot(t, v_c[:,1], label='Vy')
#     plt.plot(t, v_c[:,2], label='Vz')
#     plt.xlabel("Time (s)")
#     plt.ylabel("Velocity (m/s)")
#     plt.grid(True)
#     plt.legend()

#     # 5. CONTROL EFFORT (Normalized - Non-dimensional)
#     fig5 = plt.figure(figsize=(4.2, 3.0), dpi=300)

#     # Large smoothing window for clean trend
#     window = 35
#     control_smooth = np.convolve(
#         normalized_control,
#         np.ones(window)/window,
#         mode='same'
#     )

#     # Raw (noisy, thin)
#     plt.plot(t, normalized_control,
#             color='#1f77b4',
#             linewidth=1.0,
#             label='Raw')

#     # Clean moving average (smooth curve)
#     plt.plot(t, control_smooth,
#             color='black',
#             linestyle='--',
#             linewidth=2.5,
#             label='Moving Avg')

#     plt.xlabel("Time (s)", fontsize=12)
#     # Updated Y-label to reflect dimensionless ratio
#     plt.ylabel(r"Normalized $||\Delta u||$", fontsize=12) 
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)

#     plt.grid(True, linestyle=':', alpha=0.4)
#     plt.legend(fontsize=9, frameon=False, loc='best')

#     plt.tight_layout()
    
#     # 6. DOCKING CONE CONSTRAINT (Terminal Zoom)
#     fig6 = plt.figure(figsize=(4.2, 3.0), dpi=300)

#     # Relative vector (target - chaser)
#     rel_vec = p_c - p_t
#     rel_norm = np.linalg.norm(rel_vec, axis=1)

#     # Docking axis (example: target z-axis — change if needed)
#     a_hat = np.array([0, 0, 1])

#     # Compute angle
#     cos_theta = np.dot(rel_vec, a_hat) / (rel_norm + 1e-6)
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     theta = np.arccos(cos_theta)

#     theta_cone = np.deg2rad(30)

#     # --- Zoom last 3 seconds ---
#     t_end = t[-1]
#     mask = t >= (t_end - 3.0)

#     plt.plot(t[mask], np.rad2deg(theta[mask]),
#             color='#1f77b4',
#             linewidth=2.2,
#             label=r'$\theta(t)$')

#     plt.axhline(np.rad2deg(theta_cone),
#                 color='black',
#                 linestyle='--',
#                 linewidth=1.8,
#                 label=r'$\theta_{cone}$')

#     plt.xlabel("Time (s)", fontsize=12)
#     plt.ylabel("Cone Angle (deg)", fontsize=12)

#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)

#     plt.grid(True, linestyle=':', alpha=0.4)
#     plt.legend(fontsize=9, frameon=False)

#     plt.tight_layout()
#     plt.show()

#     return d0, df, emax, u_max_nd, td



def plot_performance(history, docking_threshold=0.1):

    t = np.array(history['t'])
    p_c = np.array(history['p_c'])
    p_t = np.array(history['p_t'])
    v_c = np.array(history['v_c'])
    act = np.array(history['action']) # Raw RPMs

    # -------------------------------------------------
    # Distance & Error
    # -------------------------------------------------
    err_vec = p_c - p_t
    dist = np.linalg.norm(err_vec, axis=1)

    # -------------------------------------------------
    # Control Effort (Non-Dimensional Thrust: T / T_hover)
    # -------------------------------------------------
    start_idx = min(len(t)-1, 50)
    hover_rpm = np.mean(act[start_idx:], axis=0)
    
    # Physics: Thrust is proportional to RPM squared
    thrust_proxy = np.sum(act**2, axis=1)
    hover_thrust_proxy = np.sum(hover_rpm**2) + 1e-6 # Add epsilon to avoid div-by-zero
    
    # Dimensionless Ratio (T / T_hover)
    nd_thrust = thrust_proxy / hover_thrust_proxy

    # -------------------------------------------------
    # Compute Quantitative Metrics
    # -------------------------------------------------
    d0 = dist[0]
    df = dist[-1]
    emax = np.max(dist)
    
    # Filter out absurd spikes for the print metric
    valid_thrust = nd_thrust[start_idx:]
    t_max_nd = np.percentile(valid_thrust, 99.5) if len(valid_thrust) > 0 else 1.0

    idx = np.where(dist < docking_threshold)[0]
    td = t[idx[0]] if len(idx) > 0 else None
    

    # Raw deviation
    act_dev = act - hover_rpm
    control_norm = np.linalg.norm(act_dev, axis=1)
    
    # Normalize by the magnitude of the hover state to make it dimensionless
    hover_norm = np.linalg.norm(hover_rpm) + 1e-6 # Add epsilon to avoid div by zero
    normalized_control = control_norm / hover_norm

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Initial Distance d0        : {d0:.4f} m")
    print(f"Final Distance df          : {df:.4f} m")
    print(f"Maximum Tracking Error     : {emax:.4f} m")
    print(f"Peak Thrust Ratio (99.5%)  : {t_max_nd:.4f}")
    print(f"Docking Time td            : {td:.4f} s" if td else "Docking Time td: Not reached")
    print("--------------------------------\n")

    # -------------------------------------------------
    # --- PLOTS ---
    # -------------------------------------------------

    # 1. 3D TRAJECTORY
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
    ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
    ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
    ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # 2. POSITION EVOLUTION
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,0], 'b-', label='X')
    plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,1], 'r-', label='Y')
    plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,2], 'k-', label='Z')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()

    # 3. TRACKING ERROR (Scaled for Paper)
    fig3 = plt.figure(figsize=(4.5, 3.4), dpi=300)
    plt.plot(t, err_vec[:,0], color='#1f77b4', linewidth=2.0, label='Err X')
    plt.plot(t, err_vec[:,1], color='#d62728', linewidth=2.0, label='Err Y')
    plt.plot(t, err_vec[:,2], color='black', linewidth=2.0, label='Err Z')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Tracking Error (m)", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(bottom=0.18) 

    # 4. VELOCITY
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(t, v_c[:,0], label='Vx')
    plt.plot(t, v_c[:,1], label='Vy')
    plt.plot(t, v_c[:,2], label='Vz')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    # 5. CONTROL EFFORT (Non-Dimensional Thrust)
    fig5 = plt.figure(figsize=(4.2, 3.0), dpi=300)

    # Large smoothing window for clean trend
    window = 35
    thrust_smooth = np.convolve(
        nd_thrust,
        np.ones(window)/window,
        mode='same'
    )

    # Raw (noisy, thin)
    plt.plot(t, nd_thrust,
            color='#1f77b4',
            linewidth=1.0,
            label='Raw Thrust')

    # Clean moving average (smooth curve)
    plt.plot(t, thrust_smooth,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Moving Avg')
            
    # Add a baseline for 1.0 (Hover Thrust)
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Hover State')

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Norm. Thrust ($T / T_{hover}$)", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    # 6. DOCKING CONE CONSTRAINT (Terminal Zoom)
    fig6 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    rel_vec = p_c - p_t
    rel_norm = np.linalg.norm(rel_vec, axis=1)
    a_hat = np.array([0, 0, 1])
    cos_theta = np.dot(rel_vec, a_hat) / (rel_norm + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_cone = np.deg2rad(30)

    t_end = t[-1]
    mask = t >= (t_end - 3.0)

    plt.plot(t[mask], np.rad2deg(theta[mask]), color='#1f77b4', linewidth=2.2, label=r'$\theta(t)$')
    plt.axhline(np.rad2deg(theta_cone), color='black', linestyle='--', linewidth=1.8, label=r'$\theta_{cone}$')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Cone Angle (deg)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()

    
    
    fig7 = plt.figure(figsize=(4.2, 3.0), dpi=300)

    # Large smoothing window for clean trend
    window = 35
    control_smooth = np.convolve(
        normalized_control,
        np.ones(window)/window,
        mode='same'
    )

    # Raw (noisy, thin)
    plt.plot(t, normalized_control,
            color='#1f77b4',
            linewidth=1.0,
            label='Raw')

    # Clean moving average (smooth curve)
    plt.plot(t, control_smooth,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Moving Avg')

    plt.xlabel("Time (s)", fontsize=12)
    # Updated Y-label to reflect dimensionless ratio
    plt.ylabel(r"Normalized $||\Delta u||$", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')

    plt.tight_layout()
    
    plt.show()
    

    return d0, df, emax, t_max_nd, td

def plot_performance(history, docking_threshold=0.15):

    t = np.array(history['t'])
    p_c = np.array(history['p_c'])
    p_t = np.array(history['p_t'])
    v_c = np.array(history['v_c'])
    act = np.array(history['action']) # Raw RPMs

    # -------------------------------------------------
    # Recompute Final Metrics on FULL 20s Data
    # -------------------------------------------------
    err_vec = p_c - p_t
    dist = np.linalg.norm(err_vec, axis=1)

    start_idx = min(len(t)-1, 50)
    hover_rpm = np.mean(act[start_idx:], axis=0)
    
    # Thrust proxy (T / T_hover)
    thrust_proxy = np.sum(act**2, axis=1)
    hover_thrust_proxy = np.sum(hover_rpm**2) + 1e-6
    nd_thrust = thrust_proxy / hover_thrust_proxy

    # Normalized Delta u
    act_dev = act - hover_rpm
    control_norm = np.linalg.norm(act_dev, axis=1)
    hover_norm = np.linalg.norm(hover_rpm) + 1e-6 
    normalized_control = control_norm / hover_norm

    # -------------------------------------------------
    # Compute Quantitative Metrics
    # -------------------------------------------------
    d0 = dist[0]
    df = dist[-1]
    emax = np.max(dist)
    
    # Find exact docking time td (but DO NOT truncate the plot data)
    idx = np.where(dist < docking_threshold)[0]
    td = t[idx[0]] if len(idx) > 0 else None
    
    valid_thrust = nd_thrust[start_idx:]
    t_max_nd = np.percentile(valid_thrust, 99.5) if len(valid_thrust) > 0 else 1.0

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Initial Distance d0        : {d0:.4f} m")
    print(f"Final Distance df          : {df:.4f} m")
    print(f"Maximum Tracking Error     : {emax:.4f} m")
    print(f"Peak Thrust Ratio (99.5%)  : {t_max_nd:.4f}")
    print(f"Docking Time td            : {td:.4f} s" if td else "Docking Time td: Not reached")
    print("--------------------------------\n")

    # Dynamic smoothing window based on data length
    window = min(35, max(1, len(t) // 5))

    # -------------------------------------------------
    # --- PLOTS ---
    # -------------------------------------------------

    # 1. 3D TRAJECTORY
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
    ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
    ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
    ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # 2. POSITION EVOLUTION
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,0], 'b-', label='X')
    plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,1], 'r-', label='Y')
    plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,2], 'k-', label='Z')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()

    # 3. TRACKING ERROR (Scaled for Paper)
    fig3 = plt.figure(figsize=(4.5, 3.4), dpi=300)
    plt.plot(t, err_vec[:,0], color='#1f77b4', linewidth=2.0, label='Err X')
    plt.plot(t, err_vec[:,1], color='#d62728', linewidth=2.0, label='Err Y')
    plt.plot(t, err_vec[:,2], color='black', linewidth=2.0, label='Err Z')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Tracking Error (m)", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(bottom=0.18) 

    # 4. VELOCITY
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(t, v_c[:,0], label='Vx')
    plt.plot(t, v_c[:,1], label='Vy')
    plt.plot(t, v_c[:,2], label='Vz')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    # 5. CONTROL EFFORT (Non-Dimensional Thrust)
    fig5 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    thrust_smooth = np.convolve(nd_thrust, np.ones(window)/window, mode='same')
    plt.plot(t, nd_thrust, color='#1f77b4', linewidth=1.0, label='Raw Thrust')
    plt.plot(t, thrust_smooth, color='black', linestyle='--', linewidth=2.5, label='Moving Avg')
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Hover State')
    if len(t) > start_idx:
        p_min, p_max = np.percentile(valid_thrust, 1), np.percentile(valid_thrust, 99)
        margin = max(0.05, (p_max - p_min) * 0.2)
        plt.ylim(max(0, p_min - margin), p_max + margin)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Norm. Thrust ($T / T_{hover}$)", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    # 6. DOCKING CONE CONSTRAINT (Terminal Zoom)
    fig6 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    rel_vec = p_c - p_t
    rel_norm = np.linalg.norm(rel_vec, axis=1)
    a_hat = np.array([0, 0, 1])
    cos_theta = np.dot(rel_vec, a_hat) / (rel_norm + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_cone = np.deg2rad(30)
    
    # To compare 20s fairly, we'll plot the entire timeline instead of zooming the last 3s
    plt.plot(t, np.rad2deg(theta), color='#1f77b4', linewidth=2.2, label=r'$\theta(t)$')
    plt.axhline(np.rad2deg(theta_cone), color='black', linestyle='--', linewidth=1.8, label=r'$\theta_{cone}$')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Cone Angle (deg)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()

    # 7. DELTA U NORMALIZED
    fig7 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    control_smooth = np.convolve(normalized_control, np.ones(window)/window, mode='same')
    plt.plot(t, normalized_control, color='#1f77b4', linewidth=1.0, label='Raw')
    plt.plot(t, control_smooth, color='black', linestyle='--', linewidth=2.5, label='Moving Avg')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Normalized $||\Delta u||$", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    plt.show()

    return d0, df, emax, t_max_nd, td





    
    
# ======================================================================
# MAIN EXECUTION (LINEAR TARGET)
# ======================================================================
def run():
    # --- FAIR COMPARISON SETUP ---
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION_SEC = 20.0
    
    CHASER_START = np.array([-2.5, 0.0, 1.5])
    P_OBS        = np.array([-1.0, 0.0, 1.25]) 
    R_OBS        = 0.4
    
    target_gen = LinearTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    init_s = target_gen.get_state(0)
    ekf.x = init_s # Init perfect
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, P_OBS, R_OBS, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, 
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, init_s[0:3]]),
        physics=Physics.PYB_DW, # Downwash
        pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=True, obstacles=False
    )
    PYB = env.getPyBulletClient()
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=R_OBS),
                      p.createVisualShape(p.GEOM_SPHERE, radius=R_OBS, rgbaColor=[1,0,0,0.2]),
                      P_OBS, physicsClientId=PYB)
    
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    state = STATE_TRACKING
    curr_traj = None
    last_plan = 0
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    # --- PHASE LOGIC ---
    docking_phase = 0 # 0: Approach Offset, 1: Terminal Docking
    DOCK_OFFSET = np.array([0, 0.0, 0.3]) # Target offset
    
    frozen = False
    freeze_pos_c = None
    freeze_pos_t = None
    
    history = {
        't': [], 
        'p_c': [], 'p_t': [], 
        'v_c': [], 'action': []
    }
    
    print("[SIM] Running Linear Target SCP Docking...")
    START = time.time()
    
    # Draw trajectory preview
    for t in np.arange(0, DURATION_SEC, 0.1):
        p.addUserDebugLine(target_gen.get_state(t)[0:3], target_gen.get_state(t+0.1)[0:3], [0.5, 0, 0.5], 2, physicsClientId=PYB)
            
    try:
            for i in range(int(DURATION_SEC * CTRL_FREQ)):
                sim_t = i / CTRL_FREQ
                
                # 0. CHECK WINDOW STATUS
                if not p.isConnected(physicsClientId=PYB):
                    print("\n[USER] Window closed. Finishing...")
                    break
                    
                # 1. FREEZE LOGIC
                if frozen:
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], freeze_pos_c, [0,0,0,1], physicsClientId=PYB)
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[1], freeze_pos_t, [0,0,0,1], physicsClientId=PYB)
                    env.render()
                    sync(i, START, env.CTRL_TIMESTEP)
                    continue
                    
                # 2. STEP PHYSICS
                obs, _, _, _, _ = env.step(action)
                true_state = target_gen.get_state(sim_t)
                
                p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
                
                # Visuals Update
                p_chaser = obs[0][0:3]
                p_target = obs[1][0:3]
                update_hull(hull_c, p_chaser, PYB)
                update_hull(hull_t, p_target, PYB)
                
                # Wind Disturbance
                wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
                p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
                p.addUserDebugLine(p_chaser, p_chaser + wind*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
                draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

                # EKF Step
                ekf.step(true_state[0:3] + np.random.normal(0, 0.01, 3))
                
                # Safety Check
                alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, P_OBS, R_OBS)
                
                # --- PHASE TRANSITION CHECK (Horizontal trigger for moving target) ---
                if docking_phase == 0:
                    # Use 2D distance so it dives the moment it catches up horizontally
                    xy_dist = np.linalg.norm(p_chaser[0:2] - p_target[0:2])
                    if xy_dist < 0.4:
                        docking_phase = 1
                        print(f"[PHASE] {sim_t:.2f}s | Switched to Terminal Phase (Diving in!)")

                # --- DOCKING COMPLETION CHECK (Wind-Relaxed) ---
                if docking_phase == 1 and np.linalg.norm(p_chaser - p_target) < 0.15:
                    frozen = True
                    freeze_pos_c = p_chaser
                    freeze_pos_t = p_target
                    
                    print(f"\n[COMPLETE] DOCKED SUCCESSFULLY!")
                    print(f">>> Total Docking Time: {sim_t:.2f} seconds <<<")
                    
                    # Capture final frame before freezing
                    history['t'].append(sim_t)
                    history['p_c'].append(obs[0][0:3].copy())
                    history['p_t'].append(true_state[0:3].copy())
                    history['v_c'].append(obs[0][10:13].copy())
                    history['action'].append(action[0].copy())
                    
                    plot_performance(history)
                    continue

                # 3. CONTROL LOGIC
                if state == STATE_TRACKING:
                    # Backoff Condition
                    if alpha_obs < ALPHA_LIMIT:
                        print(f"[ALERT] Obstacle Collision Risk! Backing Off.")
                        state = STATE_BACKING_OFF
                        backoff_start = p_chaser.copy()
                        vec = p_chaser - P_OBS
                        vec = vec / (np.linalg.norm(vec)+1e-6)
                        backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                        backoff_t_start = time.time()
                        p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
                    else:
                        # Plan
                        res = planner.get()
                        if res is not None: 
                            curr_traj = res
                            for j in range(HORIZON-1):
                                p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                        if sim_t - last_plan > PLAN_INTERVAL:
                            preds = ekf.predict_future(HORIZON, DT_PLAN)
                            if docking_phase == 0:
                                preds[:, 0:3] += DOCK_OFFSET
                            
                            chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                            planner.request(chaser_st, preds)
                            last_plan = sim_t

                        # --- THE FIX: SMOOTH BUT IMMEDIATE DIVE ---
                        # if docking_phase == 1:
                        #     # Bypass the slow MPC completely.
                        #     # Feed the raw target position AND the target's exact velocity to the PID.
                        #     # The velocity feed-forward ensures the drone smoothly glides down 
                        #     # while perfectly matching the target's forward speed.
                        #     action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_target, true_state[3:6])
                        # else:
                            # Phase 0: Follow the safe MPC trajectory to the moving offset point
                        if curr_traj is not None:
                                idx = min(3, curr_traj.shape[1]-1)
                                pt = curr_traj[0:3, idx] 
                                vt = curr_traj[3:6, idx]
                                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
                        else:
                                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START, np.zeros(3))

                elif state == STATE_BACKING_OFF:
                    elapsed = time.time() - backoff_t_start
                    progress = min(elapsed / 2.0, 1.0)
                    k = progress * progress * (3 - 2 * progress)
                    setpoint = (1-k)*backoff_start + k*backoff_end
                    action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], setpoint, np.zeros(3))
                    
                    if progress >= 1.0:
                        state = STATE_REPLANNING
                        
                elif state == STATE_REPLANNING:
                    action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser, np.zeros(3))
                    if sim_t - last_plan > 0.1:
                        preds = ekf.predict_future(HORIZON, DT_PLAN)
                        if docking_phase == 0: preds[:, 0:3] += DOCK_OFFSET
                        chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                        planner.request(chaser_st, preds)
                        last_plan = sim_t
                    
                    res = planner.get()
                    if res is not None:
                        if np.linalg.norm(res[0:3,0] - p_chaser) < 0.3:
                            curr_traj = res
                            p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                            state = STATE_TRACKING
                
                action[1] = np.zeros(4)
                
                # Print terminal output less frequently
                if i % 20 == 0:
                    dist = np.linalg.norm(p_chaser - p_target)
                    p_str = "OFFSET" if docking_phase == 0 else "TERMINAL"
                    print(f"{sim_t:05.2f} | Phase: {p_str} | Dist: {dist:.2f}m")
                
                # 4. LOGGING DATA (Only if not frozen)
                history['t'].append(sim_t)
                history['p_c'].append(obs[0][0:3].copy())
                history['p_t'].append(true_state[0:3].copy())
                history['v_c'].append(obs[0][10:13].copy())
                history['action'].append(action[0].copy())

                env.render()
                sync(i, START, env.CTRL_TIMESTEP)
            
    except Exception as e:
        print(f"\n[INFO] Simulation ended: {e}")
        
    finally:
        try: env.close()
        except: pass
        
        # Plot if time ran out without successfully docking
        if not frozen:
            print("[PLOTS] Time expired. Generating Performance Plots...")
            plot_performance(history)


if __name__ == "__main__":
    run()