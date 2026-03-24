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
        self.r_dock = 0.2 
        
        # Warm start buffers
        self.X_nom = np.zeros((self.N, 9))
        self.u_nom = np.zeros((self.N-1, 3))
        self.u_nom[:, 2] = self.GRAVITY

        self.MAX_ITERS = 4 # Lowered slightly for live simulation speed
        self.TOL = 1e-3

    def f_dyn(self, x, u):
        px, py, pz, vx, vy, vz, phi, theta, a_T = x
        phi_cmd, theta_cmd, a_cmd = u
        tau_rp, tau_t = 0.1, 0.05
        return np.array([
            vx, vy, vz,
            a_T * np.sin(theta),
            -a_T * np.sin(phi) * np.cos(theta),
            a_T * np.cos(phi) * np.cos(theta) - self.GRAVITY,
            (phi_cmd - phi) / tau_rp,
            (theta_cmd - theta) / tau_rp,
            (a_cmd - a_T) / tau_t
        ])

    def get_jacobians(self, x, u):
        px, py, pz, vx, vy, vz, phi, theta, a_T = x
        tau_rp, tau_t = 0.1, 0.05
        Ac = np.zeros((9, 9))
        Ac[0, 3] = Ac[1, 4] = Ac[2, 5] = 1.0
        Ac[3, 7] = a_T * np.cos(theta); Ac[3, 8] = np.sin(theta)
        Ac[4, 6] = -a_T * np.cos(phi) * np.cos(theta); Ac[4, 7] = a_T * np.sin(phi) * np.sin(theta); Ac[4, 8] = -np.sin(phi) * np.cos(theta)
        Ac[5, 6] = -a_T * np.sin(phi) * np.cos(theta); Ac[5, 7] = -a_T * np.cos(phi) * np.sin(theta); Ac[5, 8] = np.cos(phi) * np.cos(theta)
        Ac[6, 6] = -1.0 / tau_rp; Ac[7, 7] = -1.0 / tau_rp; Ac[8, 8] = -1.0 / tau_t
        Bc = np.zeros((9, 3))
        Bc[6, 0] = 1.0 / tau_rp; Bc[7, 1] = 1.0 / tau_rp; Bc[8, 2] = 1.0 / tau_t
        return Ac, Bc

    def solve(self, x_true, p_target, phase, is_first_step=False):
        if is_first_step:
            for k in range(self.N):
                al = k / (self.N - 1)
                self.X_nom[k, 0:3] = x_true[0:3] + al * (p_target - x_true[0:3])
                self.X_nom[k, 1] += 0.5 * np.sin(np.pi * al)
                self.X_nom[k, 8] = self.GRAVITY
        else:
            self.X_nom[:-1, :] = self.X_nom[1:, :]
            self.X_nom[-1, :] = self.X_nom[-2, :]
            self.u_nom[:-1, :] = self.u_nom[1:, :]
            self.u_nom[-1, :] = np.array([0.0, 0.0, self.GRAVITY])

        trust_radius = 2.0
        
        for it in range(self.MAX_ITERS):
            X = cp.Variable((self.N, 9))
            u = cp.Variable((self.N-1, 3))
            slack_cone = cp.Variable(self.N-1, nonneg=True)
            slack_tar  = cp.Variable(self.N-1, nonneg=True)
            nu = cp.Variable((self.N-1, 9)) 
            
            cost = 0
            con = [X[0, :] == x_true]
            offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.array([0.0, 0.0, self.r_dock])
            
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
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                trust_radius *= 0.5
                continue
                
            delta = np.linalg.norm(X.value - self.X_nom, np.inf)
            self.X_nom = X.value.copy()
            self.u_nom = u.value.copy() 
            
            if delta < self.TOL: break
            trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

        # Return the very first control command
        return self.u_nom[0, :]

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
chaser_hist = [] # Store past trajectory points

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        
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
            
            chaser_hist.append(pos.copy()) # <--- ADD THIS LINE HERE
            if len(chaser_hist) > 200: chaser_hist.pop(0) # Keep array small
            # Build 9D State for NMPC
            state_9d = np.array([
                pos[0], pos[1], pos[2], 
                lin_vel[0], lin_vel[1], lin_vel[2], 
                roll, pitch, last_thrust_state
            ])
            
            # Phase Logic
            dist_xy = np.linalg.norm(pos[0:2] - p_target_true[0:2])
            if current_phase == 0 and dist_xy < 0.2:
                current_phase = 1
                print(f"[{data.time:.1f}s] FSM TRIGGER: Phase 1 (Cone) Activated!")
                
            # Add slight sensor noise to simulate real-world targeting
            sensor_noise = np.random.normal(0, 0.01, 3) 
            p_target = p_target_true + sensor_noise

            # Solve! (This might cause a tiny stutter in the viewer depending on CPU speed)
            is_first = (last_nmpc_time < 0)
            u_opt = nmpc_planner.solve(state_9d, p_target, current_phase, is_first_step=is_first)
            
            # Extract NMPC commands
            target_roll = u_opt[0]
            target_pitch = u_opt[1]
            target_acc = u_opt[2]
            
            last_thrust_state = target_acc # Feedback for next NMPC cycle
            last_nmpc_time = data.time
            print(f"NMPC Updated at t={data.time:.2f}s | Phase: {current_phase} | Thrust Cmd: {target_acc:.2f}")

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
            cone_apex = p_target_true
            cone_length = 1.0
            cone_radius = cone_length * np.tan(np.radians(30)) # THETA = 30
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
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