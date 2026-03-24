import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time

# =====================================================================
# 1. MODULAR NMPC CLASS (Your exact SCP math, wrapped up)
# =====================================================================
class DroneNMPC:
    def __init__(self, dt=0.1, N=25):
        self.dt = dt
        self.N = N
        self.U_MAX = 15.0
        self.U_MIN = 2.0
        self.MAX_TILT = np.radians(25)
        self.V_MAX = 5.0
        self.GRAVITY = 9.81
        
        self.P_OBS = np.array([-1.0, 0.0, 1.25])
        self.R_OBS = 0.4
        self.R_SAFE = 0.1
        self.THETA = np.radians(30)
        self.N_APP = np.array([0, 0, -1])
        self.r_dock = 0
        
        # Warm start buffers upgraded to 12D
        self.X_nom = np.zeros((self.N, 12))
        self.u_nom = np.zeros((self.N-1, 3))
        self.u_nom[:, 2] = self.GRAVITY

        self.MAX_ITERS = 4 
        self.TOL = 1e-3

    def f_dyn(self, x, u):
        px, py, pz, vx, vy, vz, phi, theta, a_T, wx, wy, wz = x
        phi_cmd, theta_cmd, a_cmd = u
        tau_rp, tau_t = 0.1, 0.05
        return np.array([
            vx, vy, vz,
            a_T * np.sin(theta) + wx,                        # Wind directly pushes acceleration
            -a_T * np.sin(phi) * np.cos(theta) + wy,
            a_T * np.cos(phi) * np.cos(theta) - self.GRAVITY + wz,
            (phi_cmd - phi) / tau_rp,
            (theta_cmd - theta) / tau_rp,
            (a_cmd - a_T) / tau_t,
            0.0, 0.0, 0.0                                    # Wind is assumed constant over the short prediction horizon
        ])

    def get_jacobians(self, x, u):
        px, py, pz, vx, vy, vz, phi, theta, a_T, wx, wy, wz = x
        tau_rp, tau_t = 0.1, 0.05
        Ac = np.zeros((12, 12))
        Ac[0, 3] = Ac[1, 4] = Ac[2, 5] = 1.0
        Ac[3, 7] = a_T * np.cos(theta); Ac[3, 8] = np.sin(theta)
        Ac[4, 6] = -a_T * np.cos(phi) * np.cos(theta); Ac[4, 7] = a_T * np.sin(phi) * np.sin(theta); Ac[4, 8] = -np.sin(phi) * np.cos(theta)
        Ac[5, 6] = -a_T * np.sin(phi) * np.cos(theta); Ac[5, 7] = -a_T * np.cos(phi) * np.sin(theta); Ac[5, 8] = np.cos(phi) * np.cos(theta)
        Ac[6, 6] = -1.0 / tau_rp; Ac[7, 7] = -1.0 / tau_rp; Ac[8, 8] = -1.0 / tau_t
        Ac[3:6, 9:12] = np.eye(3) # <--- Wind Jacobian (dv/dw = I)
        
        Bc = np.zeros((12, 3))
        Bc[6, 0] = 1.0 / tau_rp; Bc[7, 1] = 1.0 / tau_rp; Bc[8, 2] = 1.0 / tau_t
        return Ac, Bc

    def solve(self, x_true, p_target, phase, is_first_step=False):
        
        delta = 0.0
        step_cost = 0.0
        
        if is_first_step:
            for k in range(self.N):
                al = k / (self.N - 1)
                self.X_nom[k, 0:3] = x_true[0:3] + al * (p_target - x_true[0:3])
                self.X_nom[k, 1] += 0.5 * np.sin(np.pi * al)
                self.X_nom[k, 8] = self.GRAVITY
                self.X_nom[k, 9:12] = x_true[9:12] # Initialize wind estimate along horizon
        else:
            self.X_nom[:-1, :] = self.X_nom[1:, :]
            self.X_nom[-1, :] = self.X_nom[-2, :]
            self.u_nom[:-1, :] = self.u_nom[1:, :]
            self.u_nom[-1, :] = np.array([0.0, 0.0, self.GRAVITY])

        trust_radius = 2.0
        
        for it in range(self.MAX_ITERS):
            X = cp.Variable((self.N, 12)) # <--- 12D NMPC Variable
            u = cp.Variable((self.N-1, 3))
            slack_cone = cp.Variable(self.N-1, nonneg=True)
            slack_tar  = cp.Variable(self.N-1, nonneg=True)
            nu = cp.Variable((self.N-1, 12)) # <--- 12D Virtual Controls
            
            cost = 0
            con = [X[0, :] == x_true]
            # offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
            offset = np.array([0.0, 0.0, 0.3]) if phase == 0 else np.array([0.0, 0.0, 0.2])
            
            
            cost += cp.sum_squares(X[-1, 0:3] - (p_target + offset)) * 1000.0
            cost += cp.sum_squares(X[-1, 3:6]) * 500.0
            cost += cp.sum_squares(X[-1, 6:8]) * 500.0
            
            for k in range(self.N-1):
                x_k_nom = self.X_nom[k, :]
                u_k_nom = self.u_nom[k, :]
                f_k = self.f_dyn(x_k_nom, u_k_nom)
                Ac, Bc = self.get_jacobians(x_k_nom, u_k_nom)
                
                con += [X[k+1, :] == X[k, :] + self.dt * (f_k + Ac @ (X[k, :] - x_k_nom) + Bc @ (u[k, :] - u_k_nom)) + nu[k, :]]
                cost += cp.sum_squares(nu[k, :]) * 1e5
                
                p_rel = X[k, 0:3] - p_target
                cost += cp.sum_squares(p_rel - offset) * 2.0 
                cost += cp.sum_squares(X[k, 3:6]) * 1.0 
                cost += cp.sum_squares(X[k, 6:8]) * 5.0 
                cost += cp.sum_squares(u[k, 0:2]) * 2.0 
                cost += cp.sum_squares(u[k, 2] - self.GRAVITY) * 0.1 
                cost += cp.sum_squares(u[k, :] - u_k_nom) * 2.0 
                
                if k > 0:
                    con += [cp.norm(X[k, :] - x_k_nom, np.inf) <= trust_radius]
                con += [cp.norm(u[k, :] - u_k_nom, np.inf) <= trust_radius]
                con += [u[k, 0] >= -self.MAX_TILT, u[k, 0] <= self.MAX_TILT] 
                con += [u[k, 1] >= -self.MAX_TILT, u[k, 1] <= self.MAX_TILT] 
                con += [u[k, 2] >= self.U_MIN, u[k, 2] <= self.U_MAX]        
                con += [cp.norm(X[k+1, 3:6], 2) <= self.V_MAX]
                
                p_rel_nom = self.X_nom[k, 0:3]
                v_obs = p_rel_nom - self.P_OBS
                d_obs = np.linalg.norm(v_obs) + 1e-8
                n_obs = v_obs / d_obs
                con += [n_obs @ (X[k, 0:3] - self.P_OBS) >= self.R_OBS + self.R_SAFE]

                if phase == 1:
                    dist_tar_nom = np.linalg.norm(p_rel_nom - p_target) + 1e-8
                    n_tar = ((p_rel_nom - p_target) / dist_tar_nom) * self.r_dock
                    con += [n_tar @ p_rel >= self.r_dock - slack_tar[k]]
                    con += [cp.norm(p_rel) * np.cos(self.THETA) <= -self.N_APP @ p_rel + slack_cone[k]]
                    con += [slack_tar[k] == 0] 
                else:
                    con += [slack_tar[k] == 0]
                    con += [slack_cone[k] == 0]

            con += [cp.norm(X[-1, :] - self.X_nom[-1, :], np.inf) <= trust_radius]
            cost += cp.sum(slack_cone) * 100.0 + cp.sum(slack_tar) * 100.0

            prob = cp.Problem(cp.Minimize(cost), con)
            prob.solve(solver=cp.CLARABEL, warm_start=True, ignore_dpp=True) 
            
            # if prob.status not in ["optimal", "optimal_inaccurate"]:
            #     print(f"⚠️ SCP WARNING: CVXPY Status = {prob.status} at Iteration {iter}")
            # break # Breaks out, meaning delta never gets calculated
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                trust_radius *= 0.5
                continue
                
            delta = np.linalg.norm(X.value - self.X_nom, np.inf)
            self.X_nom = X.value.copy()
            self.u_nom = u.value.copy() 
            
            if delta < self.TOL: break
            trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

        # Return the very first control command
        step_cost = prob.value if prob.value is not None else 0.0
        return self.u_nom[0, :], step_cost, delta, trust_radius

# =====================================================================
# 2. MUJOCO INNER LOOP CONTROLLERS
# =====================================================================
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
    def compute(self, pos_err, vel_err):
        return self.kp * pos_err + self.kd * vel_err

# =====================================================================
# 3. LIVE MUJOCO SIMULATION
# =====================================================================
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

MASS = 0.027
GRAVITY = 9.81

# We ONLY need inner loop PD to convert NMPC Angles -> Torques
pd_roll  = PDController(kp=200.0, kd=50.0) 
pd_pitch = PDController(kp=200.0, kd=50.0)
pd_yaw   = PDController(kp=100.0, kd=25.0)

# Init NMPC Planner
nmpc_planner = DroneNMPC(dt=0.1, N=25)
p_target_true = np.array([1.0, 0.0, 1.0])

# Start state
x0 = np.array([-2.5, 0.0, 1.5])
data.qpos[0:3] = x0
mujoco.mj_forward(model, data)

NMPC_DT = 0.1
last_nmpc_time = -NMPC_DT # Force immediate solve on first step
target_roll, target_pitch, target_acc = 0.0, 0.0, GRAVITY
current_phase = 0
last_thrust_state = GRAVITY 

# =====================================================================
# VISUALIZATION HELPER (Draws lines directly in the viewer)
# =====================================================================
def draw_line(viewer, pt1, pt2, rgba, width=2.0):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    # Ensure correct format
    pt1 = np.asarray(pt1, dtype=np.float64).reshape(3,)
    pt2 = np.asarray(pt2, dtype=np.float64).reshape(3,)
    rgba = np.asarray(rgba, dtype=np.float32)

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba
    )

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width),
        pt1,   # ✅ pass full vector
        pt2    # ✅ pass full vector
    )

    viewer.user_scn.ngeom += 1
    
    
    
# =====================================================================
# EKF SETUP (Directly from SCP_static_EKF.py)
# =====================================================================
P_ekf = np.eye(6) * 0.1
Q_ekf = np.eye(6) * 0.05  # Increased base noise
Q_ekf[3:6, 3:6] = np.eye(3) * 0.2  # Let the velocity adapt to reality much faster
R_kf = np.eye(3) * 0.05

A_d = np.eye(6)
A_d[0:3, 3:6] = NMPC_DT * np.eye(3)

B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * NMPC_DT**2 * np.eye(3)
B_d[3:6, :] = NMPC_DT * np.eye(3)



# ================= TARGET POSE EKF =================
# ================= TARGET POSE EKF (Upgraded to 6D to track wind drift) =================
P_tar = np.eye(6) * 0.1
Q_tar = np.eye(6) * 0.01; Q_tar[3:6, 3:6] = np.eye(3) * 0.1 # Process noise for wind drift
R_tar = np.eye(3) * 0.01

F_tar = np.eye(6)
F_tar[0:3, 3:6] = np.eye(3) * NMPC_DT

def target_predict(x_hat, P):
    return F_tar @ x_hat, F_tar @ P @ F_tar.T + Q_tar

def target_update(x_hat, P, z):
    H = np.zeros((3, 6)); H[0:3, 0:3] = np.eye(3)
    y = z - H @ x_hat
    S = H @ P @ H.T + R_tar
    K = P @ H.T @ np.linalg.inv(S)
    return x_hat + K @ y, (np.eye(6) - K @ H) @ P

x_tar_hat = np.zeros(6); x_tar_hat[0:3] = p_target_true.copy()

# ================= CHASER EKF (Upgraded to 9D to estimate Wind Acceleration) =================
P_ekf = np.eye(9) * 0.1
Q_ekf = np.eye(9) * 0.05  
Q_ekf[3:6, 3:6] = np.eye(3) * 0.2  
Q_ekf[6:9, 6:9] = np.eye(3) * 0.5  # High process noise so wind state quickly adapts

R_kf = np.eye(3) * 0.05

A_d = np.eye(9)
A_d[0:3, 3:6] = NMPC_DT * np.eye(3)
A_d[0:3, 6:9] = 0.5 * NMPC_DT**2 * np.eye(3)
A_d[3:6, 6:9] = NMPC_DT * np.eye(3)

B_d = np.zeros((9, 3))
B_d[0:3, :] = 0.5 * NMPC_DT**2 * np.eye(3)
B_d[3:6, :] = NMPC_DT * np.eye(3)

def ekf_predict(x_hat, P, u_acc):
    x_hat_next = A_d @ x_hat + B_d @ u_acc
    P_next = A_d @ P @ A_d.T + Q_ekf
    return x_hat_next, P_next

def ekf_update(x_hat, P, z):
    H = np.zeros((3, 9)); H[0:3, 0:3] = np.eye(3)
    y = z - H @ x_hat
    S = H @ P @ H.T + R_kf
    K = P @ H.T @ np.linalg.inv(S)
    x_hat_next = x_hat + K @ y
    P_next = (np.eye(9) - K @ H) @ P
    return x_hat_next, P_next

# Initialize EKF state [px, py, pz, vx, vy, vz, wx, wy, wz]
x_hat = np.zeros(9); x_hat[0:3] = np.array([-2.5, 0.0, 1.5]) 
u_acc_prev = np.zeros(3) 

# Simulation Variables for Wind Integration
wind_true = np.zeros(3)
v_target_true = np.zeros(3)

chaser_hist = [] # Store past trajectory points


# --- LOGGING ARRAYS FOR PLOTS ---
x_hist = []
tar_hist = []
tar_est_hist = []
u_hist = []
time_steps_log = []
cost_history = []
delta_history = []
trust_history = []

r_c = 0.1
r_t = 0.1
alpha_min = 1.05



paused = False

def key_callback(keycode):
    global paused
    if keycode == 32:  # SPACE
        paused = not paused
        print("SIMULATION PAUSED" if paused else "SIMULATION RESUMED")


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    
    
    # --- DOCKING TERMINATION CRITERIA ---
    DOCK_DIST_TOL = 0.21     # meters (Max acceptable distance to be considered "docked")
    DOCK_VEL_TOL = 0.15       # m/s (Max velocity, ensures the drone has actually stopped)
    DOCK_STABILITY_TIME = 0.5 # seconds (Must hold position for this long to prove stability)
    MAX_SIM_TIME = 30.0       # seconds (Failsafe timeout if it gets stuck)

    stable_dock_timer = 0.0
    
    while viewer.is_running():
        step_start = time.time()
        
        if not paused:
        
            # --- 0. WIND DISTURBANCE PHYSICS (Random Walk) ---
            wind_true += np.random.normal(0, 0.1, 3) * model.opt.timestep
            wind_true = np.clip(wind_true, -2.5, 2.5) # Limit extreme hurricane winds
            
            # Apply wind physically to the Target
            v_target_true = v_target_true * 0.995 + wind_true * model.opt.timestep # 0.995 is slight drag
            p_target_true += v_target_true * model.opt.timestep
            
            target_mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target") # Change "target" to whatever it's named in your XML
            data.mocap_pos[target_mocap_id] = p_target_true
            
            # Apply wind physically to the Chaser (External Force mapping)
            data.qfrc_applied[0:3] = MASS * wind_true
            
            # --- 1. SENSOR READINGS ---
            pos = data.qpos[0:3]
            quat = data.qpos[3:7] 
            lin_vel = data.qvel[0:3] 
            ang_vel = data.qvel[3:6] 
            
            w, xq, yq, zq = quat
            roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
            pitch = np.arcsin(2*(w*yq - zq*xq))
            yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))

            # --- 2. NMPC OUTER LOOP (Runs at 10Hz) ---
            if data.time - last_nmpc_time >= NMPC_DT:
                chaser_hist.append(pos.copy()) 
                if len(chaser_hist) > 200: chaser_hist.pop(0) 
                
                # ====================================================
                # 1. TARGET ESTIMATION (6D EKF) - DO THIS FIRST!
                # ====================================================
                z_tar = p_target_true + np.random.normal(0, 0.01, 3)
                x_tar_hat, P_tar = target_predict(x_tar_hat, P_tar)
                x_tar_hat, P_tar = target_update(x_tar_hat, P_tar, z_tar)
                p_tar_hat = x_tar_hat[0:3]

                # ====================================================
                # 2. CHASER ESTIMATION (9D EKF with Wind)
                # ====================================================
                z_cha = pos + np.random.normal(0, 0.01, 3)
                x_hat, P_ekf = ekf_predict(x_hat, P_ekf, u_acc_prev)
                x_hat, P_ekf = ekf_update(x_hat, P_ekf, z_cha)
                
                # Build 12D State for NMPC using EKF estimates (Pos/Vel/Wind)
                state_12d = np.array([
                    x_hat[0], x_hat[1], x_hat[2],   # px, py, pz
                    x_hat[3], x_hat[4], x_hat[5],   # vx, vy, vz
                    roll, pitch, last_thrust_state, # roll, pitch, thrust
                    x_hat[6], x_hat[7], x_hat[8]    # wx, wy, wz (Live Wind Estimate)
                ])
                
                # ====================================================
                # 3. PHASE LOGIC & NMPC SOLVE
                # ====================================================
                # FSM triggers based strictly on ESTIMATED positions
                dist_xy = np.linalg.norm(x_hat[0:2] - p_tar_hat[0:2])
                if current_phase == 0 and dist_xy < 0.15:
                    current_phase = 1
                    print(f"[{data.time:.1f}s] FSM TRIGGER: Phase 1 (Cone) Activated!")

                # Solve NMPC using 12D State
                is_first = (last_nmpc_time < 0)
                u_opt, step_cost, step_delta, step_trust = nmpc_planner.solve(
                    state_12d, p_tar_hat, current_phase, is_first_step=is_first
                )
                
                # Extract NMPC commands
                target_roll = u_opt[0]
                target_pitch = u_opt[1]
                target_acc = u_opt[2]
                
                # ====================================================
                # 4. LOGGING & STATE PREP FOR NEXT LOOP
                # ====================================================
                x_hist.append(state_12d.copy())
                tar_hist.append(p_target_true.copy())
                tar_est_hist.append(p_tar_hat.copy())
                u_hist.append(u_opt.copy())
                time_steps_log.append(data.time)
                cost_history.append(step_cost)
                delta_history.append(step_delta)
                trust_history.append(step_trust)
                
                # Convert NMPC [roll, pitch, thrust] into Cartesian [ax, ay, az] for the next EKF Predict step
                u_acc_prev = np.array([
                    target_acc * np.sin(target_pitch),
                    -target_acc * np.sin(target_roll) * np.cos(target_pitch),
                    target_acc * np.cos(target_roll) * np.cos(target_pitch) - GRAVITY
                ])
                
                last_thrust_state = target_acc 
                last_nmpc_time = data.time
                print(f"t={data.time:.1f}s | Ph: {current_phase} | Est. Wind: [{x_hat[6]:.2f}, {x_hat[7]:.2f}] m/s²")

                # =========================================================
                # 🛑 DOCKING TERMINATION CHECK
                # =========================================================
                dist_3d = np.linalg.norm(x_hat[0:3] - x_tar_hat[0:3])
                vel_norm = np.linalg.norm(x_hat[3:6])
                
                if current_phase == 1 and dist_3d < DOCK_DIST_TOL and vel_norm < DOCK_VEL_TOL:
                    stable_dock_timer += NMPC_DT
                    if stable_dock_timer >= DOCK_STABILITY_TIME:
                        print(f"\n✅ DOCKING SUCCESSFUL!")
                        print(f"⏱️ Time to Dock: {data.time:.2f} seconds.")
                        print(f"📏 Final Error: {dist_3d:.3f} meters.")
                        break  # This exits the MuJoCo loop and opens your plots!
                else:
                    stable_dock_timer = 0.0 # Reset if it drifts out of tolerance
                    
                if data.time >= MAX_SIM_TIME:
                    print(f"\n❌ SIMULATION TIMEOUT: Reached {MAX_SIM_TIME}s without stable docking.")
                    break
                # =========================================================
                
                
                
                # Print the live wind estimate so you can watch it learn!
                print(f"NMPC Updated at t={data.time:.2f}s | Phase: {current_phase} | Est. Wind: [{x_hat[6]:.2f}, {x_hat[7]:.2f}, {x_hat[8]:.2f}] m/s²")
            # --- 3. MUJOCO INNER LOOP (Runs at 500Hz) ---
            # Convert NMPC Target Angles to Motor Torques
            err_roll  = target_roll - roll
            err_pitch = target_pitch - pitch
            err_yaw   = 0.0 - yaw
            
            u_roll  = pd_roll.compute(err_roll,  0.0 - ang_vel[0])
            u_pitch = pd_pitch.compute(err_pitch, 0.0 - ang_vel[1])
            u_yaw   = pd_yaw.compute(err_yaw,    0.0 - ang_vel[2])
            
            # Convert NMPC Target Acceleration to physical Force
            total_thrust = MASS * target_acc
            
            data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
            data.ctrl[1] = -u_roll
            data.ctrl[2] = -u_pitch
            data.ctrl[3] = -u_yaw

            mujoco.mj_step(model, data)
            
            
            # --- 4. HOLOGRAPHIC VISUALIZATIONS ---
            with viewer.lock():
                viewer.user_scn.ngeom = 0 # Clear old lines
                
                # A. Draw Past Trajectory (Solid Red Line)
                for i in range(len(chaser_hist)-1):
                    draw_line(viewer, chaser_hist[i], chaser_hist[i+1], np.array([1, 0, 0, 1]), width=2)
                    
                # B. Draw SCP Lookahead Trajectory (Flickering Green Line)
                # nmpc_planner.X_nom contains the 25-step future path
                lookahead = nmpc_planner.X_nom[:, 0:3]
                for i in range(len(lookahead)-1):
                    draw_line(viewer, lookahead[i], lookahead[i+1], np.array([0, 1, 0, 1]), width=4)

                # C. Draw Proper Wireframe Docking Cone (Cyan)
                # cone_apex = p_target_true
                
                final_target_pos = tar_hist[-1]
                cone_apex = final_target_pos
                
                cone_length = 1.0
                cone_radius = cone_length * np.tan(np.radians(30)) # THETA = 30
                for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
                    # base_pt = cone_apex + np.array([cone_radius*np.cos(angle), cone_radius*np.sin(angle), -cone_length])
                    base_pt = cone_apex + np.array([
                            cone_radius*np.cos(angle),
                            cone_radius*np.sin(angle),
                            cone_length   # ✅ NOW it goes upward
                        ])
                    draw_line(viewer, cone_apex, base_pt, np.array([0, 1, 1, 0.4]), width=2)

        viewer.sync()

        # Keep sync
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
            
# ====================================================================
# IEEE PUBLICATION PLOTS (REVISED & AUTO-SAVING)
# ====================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Create a folder for the plots so they don't clutter your main directory
os.makedirs("ieee_plots", exist_ok=True)

x_hist = np.array(x_hist)
tar_hist = np.array(tar_hist)
tar_est_hist = np.array(tar_est_hist)
u_hist = np.array(u_hist)
time_steps = np.array(time_steps_log)

U_MAX = nmpc_planner.U_MAX
U_MIN = nmpc_planner.U_MIN
MAX_TILT = nmpc_planner.MAX_TILT
V_MAX = nmpc_planner.V_MAX
P_OBS = nmpc_planner.P_OBS
R_OBS = nmpc_planner.R_OBS
R_SAFE = nmpc_planner.R_SAFE
N_APP = nmpc_planner.N_APP
THETA = nmpc_planner.THETA
TOL = nmpc_planner.TOL

# Enforce IEEE Publication Standards (Large fonts, Serif, No Titles, No Fills)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#b0b0b0',
    'axes.titlesize': 0  
})

SINGLE_COL = (4.0, 3.2) 
traj = x_hist[:, 0:3]
dock_time = time_steps[-1]

# ⭐ NEW PLOT: Target Estimation Performance
fig_est = plt.figure(figsize=SINGLE_COL)
ax_est = plt.gca()
est_error = np.linalg.norm(tar_hist - tar_est_hist, axis=1)
ax_est.plot(time_steps, est_error, 'k-', linewidth=2, label='$||p_{target} - \hat{p}_{target}||$')
ax_est.set_xlabel('Time (s)')
ax_est.set_ylabel('Estimation Error (m)')
ax_est.legend()
plt.tight_layout()
fig_est.savefig("ieee_plots/0_target_estimation_error.png", dpi=300, bbox_inches='tight')

# ✅ FIG 1: 3D Trajectory (Main Result)
fig1 = plt.figure(figsize=(4.5, 4.5))
ax1 = fig1.add_subplot(111, projection='3d')
# Faded nominal straight-line path
ax1.plot([traj[0,0], tar_hist[0,0]], [traj[0,1], tar_hist[0,1]], [traj[0,2], tar_hist[0,2]], 
         color='gray', linestyle=':', linewidth=1.5, label='Nominal Direct Path')
# Thicker chaser line
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', linewidth=3.5, label='Executed Chaser Traj')
ax1.plot(tar_hist[:,0], tar_hist[:,1], tar_hist[:,2], 'r-', linewidth=2, label='True Moving Target')
ax1.plot([traj[0,0]], [traj[0,1]], [traj[0,2]], 'go', markersize=6, label='Start')
# Docking point marker
ax1.plot([tar_hist[-1,0]], [tar_hist[-1,1]], [tar_hist[-1,2]], 'r*', markersize=10, label='Docking Point')

# Semi-transparent obstacle
u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.15, edgecolor='none')

# Static snapshot of Cone at final docking location
cone_apex = tar_hist[-1]
cone_length = 1.0
cone_radius = cone_length * np.tan(THETA)
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    base_pt = cone_apex + np.array([cone_radius*np.cos(angle), cone_radius*np.sin(angle), cone_length])
    ax1.plot([cone_apex[0], base_pt[0]], [cone_apex[1], base_pt[1]], [cone_apex[2], base_pt[2]], 'c-', alpha=0.6)

ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1.05))
plt.tight_layout()
fig1.savefig("ieee_plots/1_3D_trajectory.png", dpi=300, bbox_inches='tight')

# ✅ FIG 2: Control Inputs (Split: Tilt & Thrust)
fig2, (ax_tilt, ax_thr) = plt.subplots(2, 1, figsize=(4.0, 4.5), sharex=True)
# Subplot 1: Tilt
ax_tilt.plot(time_steps, np.degrees(u_hist[:, 0]), 'r-', label='Roll Cmd ($\\phi$)')
ax_tilt.plot(time_steps, np.degrees(u_hist[:, 1]), 'g-', label='Pitch Cmd ($\\theta$)')
ax_tilt.axhline(np.degrees(MAX_TILT), color='k', linestyle='--', linewidth=1.5, label='Tilt Limit')
ax_tilt.axhline(-np.degrees(MAX_TILT), color='k', linestyle='--', linewidth=1.5)
ax_tilt.set_ylabel('Tilt Angle (deg)')
ax_tilt.legend(loc='upper right')
# Subplot 2: Thrust
ax_thr.plot(time_steps, u_hist[:, 2], 'b-', label='Thrust Cmd ($a_T$)')
ax_thr.axhline(U_MAX, color='k', linestyle='--', linewidth=1.5, label='Max Thrust')
ax_thr.axhline(U_MIN, color='k', linestyle='--', linewidth=1.5, label='Min Thrust')
ax_thr.set_xlabel('Time (s)')
ax_thr.set_ylabel('Thrust ($m/s^2$)')
ax_thr.legend(loc='lower right')
plt.tight_layout()
fig2.savefig("ieee_plots/2_control_inputs.png", dpi=300, bbox_inches='tight')

# ✅ FIG 3: Velocity Norm
fig3 = plt.figure(figsize=SINGLE_COL)
ax_v = plt.gca()
v_norms = np.linalg.norm(x_hist[:, 3:6], axis=1)
ax_v.plot(time_steps, v_norms, 'purple', linewidth=2.5, label='$||v||_2$')
ax_v.axhline(V_MAX, color='k', linestyle='--', linewidth=2, label='$V_{max}$ Constraint')
ax_v.set_xlabel('Time (s)')
ax_v.set_ylabel('Velocity Magnitude ($m/s$)')
ax_v.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.tight_layout()
fig3.savefig("ieee_plots/3_velocity_norm.png", dpi=300, bbox_inches='tight')

# ✅ FIG 4: Obstacle Distance
fig4 = plt.figure(figsize=SINGLE_COL)
ax_obs = plt.gca()
dist_obs = np.linalg.norm(traj - P_OBS, axis=1)
min_dist = np.min(dist_obs)
min_idx = np.argmin(dist_obs)

ax_obs.plot(time_steps, dist_obs, 'r-', linewidth=2)
# Bold boundary, NO fill
ax_obs.axhline(R_OBS + R_SAFE, color='k', linestyle='-', linewidth=2.5, label=f'Safe Boundary ({R_OBS + R_SAFE}m)')
# Marker for min distance
ax_obs.plot(time_steps[min_idx], min_dist, 'ko', markersize=6, label=f'Min Dist: {min_dist:.2f}m')
ax_obs.set_xlabel('Time (s)')
ax_obs.set_ylabel('Distance to Obstacle (m)')
ax_obs.legend()
plt.tight_layout()
fig4.savefig("ieee_plots/4_obstacle_distance.png", dpi=300, bbox_inches='tight')

# ✅ FIG 6: Docking Cone Angle
fig6 = plt.figure(figsize=SINGLE_COL)
ax_cone = plt.gca()
angles = []
for p_c, p_t in zip(traj, tar_hist):
    p_rel = p_c - p_t
    dist = np.linalg.norm(p_rel)
    if dist < 1e-5:
        angles.append(0.0)
    else:
        cos_phi = np.clip(np.dot(-N_APP, p_rel) / dist, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_phi)))
        
ax_cone.plot(time_steps, angles, 'c-', linewidth=2, label='Approaching Angle')
# Highlight violation limit (Red dashed)
ax_cone.axhline(np.degrees(THETA), color='r', linestyle='--', linewidth=2, label=f'Violation Limit ({30}°)')
# Docking time annotation
ax_cone.axvline(dock_time, color='k', linestyle=':', linewidth=1.5, label='Docking Achieved')

ax_cone.set_xlabel('Time (s)')
ax_cone.set_ylabel('Approach angle $\\phi$ (deg)')
ax_cone.legend(loc="upper right")
plt.tight_layout()
fig6.savefig("ieee_plots/6_docking_cone_angle.png", dpi=300, bbox_inches='tight')

# ✅ FIG 8: Distance to Target (Interception Performance)
fig8 = plt.figure(figsize=SINGLE_COL)
ax_r = plt.gca()
dist_to_target = np.linalg.norm(traj - tar_hist, axis=1)
ax_r.plot(time_steps, dist_to_target, 'm-', linewidth=2.5, label='Relative Distance')
ax_r.axhline(0.2, color='k', linestyle='--', linewidth=1.5, label='Docking Tolerance (0.2m)')

# Annotate convergence rate
# ax_r.annotate('Exponential Convergence', xy=(time_steps[len(time_steps)//2], dist_to_target[len(time_steps)//2]), 
#              xytext=(time_steps[len(time_steps)//3], dist_to_target[len(time_steps)//3] + 0.5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

ax_r.set_xlabel('Time (s)')
ax_r.set_ylabel('Distance to Target (m)')
ax_r.legend()
plt.tight_layout()
fig8.savefig("ieee_plots/8_distance_to_target.png", dpi=300, bbox_inches='tight')

# ----------------- Solver Metrics (Cost, Trust Region, Delta) -----------------
fig_metrics, (ax_c, ax_d, ax_t) = plt.subplots(3, 1, figsize=(4.0, 6.0), sharex=True)
iters = range(1, len(cost_history)+1)

ax_c.plot(iters, cost_history, 'm-', linewidth=2)
ax_c.set_ylabel('Objective Cost')

ax_d.semilogy(iters, delta_history, 'c-', linewidth=2)
ax_d.axhline(TOL, color='k', linestyle='--', linewidth=1.5)
ax_d.set_ylabel('Max Change $\\delta$')

ax_t.plot(iters, trust_history, 'y-', linewidth=2)
ax_t.set_xlabel('NMPC Step')
ax_t.set_ylabel('Trust Radius (m)')
plt.tight_layout()
fig_metrics.savefig("ieee_plots/9_solver_metrics.png", dpi=300, bbox_inches='tight')

plt.show()