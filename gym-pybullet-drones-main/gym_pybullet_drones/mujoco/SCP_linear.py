import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time

# =====================================================================
# 1. MODULAR NMPC CLASS (Upgraded for Moving Target Projection)
# =====================================================================
class DroneNMPC:
    def __init__(self, dt=0.1, N=25):
        self.dt = dt
        self.N = N
        self.U_MAX = 15.
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
        
        self.X_nom = np.zeros((self.N, 9))
        self.u_nom = np.zeros((self.N-1, 3))
        self.u_nom[:, 2] = self.GRAVITY

        self.MAX_ITERS = 4 
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

    def solve(self, x_true, target_est, phase, is_first_step=False):
        tar_pos = target_est[0:3]
        tar_vel = target_est[3:6]
        
        if is_first_step:
            for k in range(self.N):
                al = k / (self.N - 1)
                p_tar_k = tar_pos + k * self.dt * tar_vel
                self.X_nom[k, 0:3] = x_true[0:3] + al * (p_tar_k - x_true[0:3])
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
            # offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
            offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.array([0.0, 0.0, 0.2])
            
            
            # Match Final Projected Target Position AND Target Velocity
            p_tar_N = tar_pos + (self.N - 1) * self.dt * tar_vel
            cost += cp.sum_squares(X[-1, 0:3] - (p_tar_N + offset)) * 1000.0
            cost += cp.sum_squares(X[-1, 3:6] - tar_vel) * 500.0 
            cost += cp.sum_squares(X[-1, 6:8]) * 500.0
            
            for k in range(self.N-1):
                x_k_nom = self.X_nom[k, :]
                u_k_nom = self.u_nom[k, :]
                f_k = self.f_dyn(x_k_nom, u_k_nom)
                Ac, Bc = self.get_jacobians(x_k_nom, u_k_nom)
                
                con += [X[k+1, :] == X[k, :] + self.dt * (f_k + Ac @ (X[k, :] - x_k_nom) + Bc @ (u[k, :] - u_k_nom)) + nu[k, :]]
                cost += cp.sum_squares(nu[k, :]) * 1e5
                
                # Dynamic Projection of the Target Coordinate at step k
                p_tar_k = tar_pos + k * self.dt * tar_vel
                p_rel = X[k, 0:3] - p_tar_k
                
                cost += cp.sum_squares(p_rel - offset) * 2.0 
                cost += cp.sum_squares(X[k, 3:6] - tar_vel) * 1.0 # Force smooth velocity matching
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
                
                # Obstacle Avoidance (Static)
                p_rel_nom = self.X_nom[k, 0:3]
                v_obs = p_rel_nom - self.P_OBS
                d_obs = np.linalg.norm(v_obs) + 1e-8
                n_obs = v_obs / d_obs
                con += [n_obs @ (X[k, 0:3] - self.P_OBS) >= self.R_OBS + self.R_SAFE]

                # Moving Docking Cone
                if phase == 1:
                    dist_tar_nom = np.linalg.norm(p_rel_nom - p_tar_k) + 1e-8
                    n_tar = ((p_rel_nom - p_tar_k) / dist_tar_nom) * self.r_dock
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

        return self.u_nom[0, :]


# =====================================================================
# 2. DUAL EKF CLASSES (Chaser & Target)
# =====================================================================
class TargetEKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(9) # [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]
        self.x[0:3] = [0.5, 0.0, 1.0] # Initial guess
        self.P = np.eye(9) * 1.0 
        
        self.F = np.eye(9)
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.F[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        self.F[3:6, 6:9] = np.eye(3) * dt
        
        self.Q = np.eye(9) * 0.001
        self.Q[6:9, 6:9] = np.eye(3) * 0.05 
        
        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.01

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.x

# Chaser EKF Functions
P_ekf = np.eye(6) * 0.1
Q_ekf = np.eye(6) * 0.05  
Q_ekf[3:6, 3:6] = np.eye(3) * 0.2  
R_kf = np.eye(3) * 0.01

NMPC_DT = 0.1
A_d = np.eye(6)
A_d[0:3, 3:6] = NMPC_DT * np.eye(3)
B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * NMPC_DT**2 * np.eye(3)
B_d[3:6, :] = NMPC_DT * np.eye(3)

def ekf_predict(x_hat, P, u_acc):
    return A_d @ x_hat + B_d @ u_acc, A_d @ P @ A_d.T + Q_ekf

def ekf_update(x_hat, P, z):
    H = np.zeros((3, 6))
    H[0:3, 0:3] = np.eye(3)
    S = H @ P @ H.T + R_kf
    K = P @ H.T @ np.linalg.inv(S)
    return x_hat + K @ (z - H @ x_hat), (np.eye(6) - K @ H) @ P


# =====================================================================
# 3. MUJOCO INNER LOOP CONTROLLERS & VISUALIZER
# =====================================================================
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp; self.kd = kd
    def compute(self, pos_err, vel_err):
        return self.kp * pos_err + self.kd * vel_err

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
# 4. LIVE SIMULATION LOOP
# =====================================================================
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

MASS = 0.027
GRAVITY = 9.81

pd_roll  = PDController(kp=200.0, kd=50.0) 
pd_pitch = PDController(kp=200.0, kd=50.0)
pd_yaw   = PDController(kp=100.0, kd=25.0)

nmpc_planner = DroneNMPC(dt=NMPC_DT, N=25)
target_ekf = TargetEKF(dt=NMPC_DT)

# True Target State & Linear Velocity (Mocap moves at 0.15 m/s diagonally)
p_target_true = np.array([1.0, 0.0, 1.0])
v_target_true = np.array([0.15, 0.1, 0.0]) 

# Chaser start state
x_hat = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
u_acc_prev = np.zeros(3)
data.qpos[0:3] = x_hat[0:3]
mujoco.mj_forward(model, data)

last_nmpc_time = -NMPC_DT 
target_roll, target_pitch, target_acc = 0.0, 0.0, GRAVITY
current_phase = 0
last_thrust_state = GRAVITY 

chaser_hist = []
target_hist = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        
        # --- 1. MOVE TARGET PHYSICALLY ---
        p_target_true += v_target_true * model.opt.timestep
        data.mocap_pos[0] = p_target_true  # Update mocap body in MuJoCo!

        # --- 2. SENSORS ---
        pos = data.qpos[0:3]
        quat = data.qpos[3:7] 
        ang_vel = data.qvel[3:6] 
        w, xq, yq, zq = quat
        roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
        pitch = np.arcsin(2*(w*yq - zq*xq))
        yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))

        # --- 3. DUAL SENSOR FUSION & NMPC LOOP (10Hz) ---
        if data.time - last_nmpc_time >= NMPC_DT:
            chaser_hist.append(pos.copy()) 
            if len(chaser_hist) > 200: chaser_hist.pop(0) 
            
            target_hist.append(p_target_true.copy())
            if len(target_hist) > 200:
                target_hist.pop(0)
            
            # Target EKF
            z_tar = p_target_true + np.random.normal(0, 0.01, 3)
            target_ekf.predict()
            target_est = target_ekf.update(z_tar)
            
            # Chaser EKF
            z_cha = pos + np.random.normal(0, 0.01, 3)
            x_hat, P_ekf = ekf_predict(x_hat, P_ekf, u_acc_prev)
            x_hat, P_ekf = ekf_update(x_hat, P_ekf, z_cha)
            
            # NMPC 9D State
            state_9d = np.array([
                x_hat[0], x_hat[1], x_hat[2], 
                x_hat[3], x_hat[4], x_hat[5], 
                roll, pitch, last_thrust_state
            ])
            
            # Phase Logic against moving target
            dist_xy = np.linalg.norm(x_hat[0:2] - target_est[0:2])
            if current_phase == 0 and dist_xy < 0.2:
                current_phase = 1
                print(f"[{data.time:.1f}s] FSM TRIGGER: Phase 1 (Cone) Activated!")

            # Solve! 
            is_first = (last_nmpc_time < 0)
            u_opt = nmpc_planner.solve(state_9d, target_est, current_phase, is_first_step=is_first)
            
            target_roll, target_pitch, target_acc = u_opt[0], u_opt[1], u_opt[2]
            u_acc_prev = np.array([
                target_acc * np.sin(target_pitch),
                -target_acc * np.sin(target_roll) * np.cos(target_pitch),
                target_acc * np.cos(target_roll) * np.cos(target_pitch) - GRAVITY
            ])
            
            last_thrust_state = target_acc 
            last_nmpc_time = data.time

        # --- 4. HOLOGRAPHIC VISUALIZATIONS ---
        with viewer.lock():
            viewer.user_scn.ngeom = 0 
            
            # Past Trajectory (Solid Red)
            for i in range(len(chaser_hist)-1):
                draw_line(viewer, chaser_hist[i], chaser_hist[i+1], np.array([1, 0, 0, 1]), width=2)
                
            # Target Past Trajectory (Solid Blue)
            for i in range(len(target_hist)-1):
                draw_line(
                    viewer,
                    target_hist[i],
                    target_hist[i+1],
                    np.array([1, 1, 0, 1]),
                    width=3
                )
                
            # SCP Lookahead (Flickering Green targeting the Moving target!)
            lookahead = nmpc_planner.X_nom[:, 0:3]
            for i in range(len(lookahead)-1):
                draw_line(viewer, lookahead[i], lookahead[i+1], np.array([0, 1, 0, 1]), width=4)

            # Moving Docking Cone (Tracks the true target drone dynamically)
            cone_apex = p_target_true
            cone_length = 1.0
            cone_radius = cone_length * np.tan(np.radians(30)) 
            for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
                base_pt = cone_apex + np.array([
                        cone_radius*np.cos(angle),
                        cone_radius*np.sin(angle),
                        cone_length   
                    ])
                draw_line(viewer, cone_apex, base_pt, np.array([0, 1, 1, 0.4]), width=2)

        # --- 5. MUJOCO INNER LOOP PD ---
        u_roll  = pd_roll.compute(target_roll - roll,  0.0 - ang_vel[0])
        u_pitch = pd_pitch.compute(target_pitch - pitch, 0.0 - ang_vel[1])
        u_yaw   = pd_yaw.compute(0.0 - yaw, 0.0 - ang_vel[2])
        
        total_thrust = MASS * target_acc
        data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
        data.ctrl[1] = -u_roll
        data.ctrl[2] = -u_pitch
        data.ctrl[3] = -u_yaw

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0: time.sleep(time_until_next_step)