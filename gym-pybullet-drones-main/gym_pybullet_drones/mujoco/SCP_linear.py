import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# =====================================================================
# 1. 12D MODULAR NMPC CLASS (Moving Target + Wind Robustness)
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
        
        # 12D Warm start buffers
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
            0.0, 0.0, 0.0                                    # Wind assumed constant over prediction horizon
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
        Ac[3:6, 9:12] = np.eye(3) # Wind Jacobian
        
        Bc = np.zeros((12, 3))
        Bc[6, 0] = 1.0 / tau_rp; Bc[7, 1] = 1.0 / tau_rp; Bc[8, 2] = 1.0 / tau_t
        return Ac, Bc

    def solve(self, state_12d, target_est, phase, is_first_step=False):
        # Prevent UnboundLocalError
        delta = 0.0
        step_cost = 0.0
        trust_radius = 2.0
        
        tar_pos = target_est[0:3]
        tar_vel = target_est[3:6]
        
        if is_first_step:
            for k in range(self.N):
                al = k / (self.N - 1)
                p_tar_k = tar_pos + k * self.dt * tar_vel
                self.X_nom[k, 0:3] = state_12d[0:3] + al * (p_tar_k - state_12d[0:3])
                self.X_nom[k, 1] += 0.5 * np.sin(np.pi * al)
                self.X_nom[k, 8] = self.GRAVITY
                self.X_nom[k, 9:12] = state_12d[9:12] # Initialize wind estimate along horizon
        else:
            self.X_nom[:-1, :] = self.X_nom[1:, :]
            self.X_nom[-1, :] = self.X_nom[-2, :]
            self.u_nom[:-1, :] = self.u_nom[1:, :]
            self.u_nom[-1, :] = np.array([0.0, 0.0, self.GRAVITY])
            # Force wind update across the nominal trajectory to prevent infeasibility
            self.X_nom[:, 9:12] = state_12d[9:12]

        for it in range(self.MAX_ITERS):
            X = cp.Variable((self.N, 12))
            u = cp.Variable((self.N-1, 3))
            slack_cone = cp.Variable(self.N-1, nonneg=True)
            slack_tar  = cp.Variable(self.N-1, nonneg=True)
            nu = cp.Variable((self.N-1, 12)) 
            
            cost = 0
            con = [X[0, :] == state_12d]
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
                print(f"⚠️ SCP WARNING: CVXPY Status = {prob.status} at Iteration {it}")
                trust_radius *= 0.5
                continue
                
            delta = np.linalg.norm(X.value - self.X_nom, np.inf)
            self.X_nom = X.value.copy()
            self.u_nom = u.value.copy() 
            step_cost = prob.value if prob.value is not None else 0.0
            
            if delta < self.TOL: break
            trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

        return self.u_nom[0, :], step_cost, delta, trust_radius


# =====================================================================
# 2. DUAL EKF CLASSES (Chaser & Target)
# =====================================================================
NMPC_DT = 0.1

# ================= TARGET EKF (6D: Pos, Vel) =================
P_tar = np.eye(6) * 0.1
Q_tar = np.eye(6) * 0.01; Q_tar[3:6, 3:6] = np.eye(3) * 0.1 
R_tar = np.eye(3) * 0.01
F_tar = np.eye(6); F_tar[0:3, 3:6] = np.eye(3) * NMPC_DT

def target_predict(x_hat, P):
    return F_tar @ x_hat, F_tar @ P @ F_tar.T + Q_tar

def target_update(x_hat, P, z):
    H = np.zeros((3, 6)); H[0:3, 0:3] = np.eye(3)
    y = z - H @ x_hat
    S = H @ P @ H.T + R_tar
    K = P @ H.T @ np.linalg.inv(S)
    return x_hat + K @ y, (np.eye(6) - K @ H) @ P

# ================= CHASER EKF (9D: Pos, Vel, Wind) =================
P_ekf = np.eye(9) * 0.1
Q_ekf = np.eye(9) * 0.05  
Q_ekf[3:6, 3:6] = np.eye(3) * 0.1  
Q_ekf[6:9, 6:9] = np.eye(3) * 0.005 # Smooth Wind Estimator

R_kf = np.eye(3) * 0.05

A_d = np.eye(9)
A_d[0:3, 3:6] = NMPC_DT * np.eye(3)
A_d[0:3, 6:9] = 0.5 * NMPC_DT**2 * np.eye(3)
A_d[3:6, 6:9] = NMPC_DT * np.eye(3)

B_d = np.zeros((9, 3))
B_d[0:3, :] = 0.5 * NMPC_DT**2 * np.eye(3)
B_d[3:6, :] = NMPC_DT * np.eye(3)

def ekf_predict(x_hat, P, u_acc):
    return A_d @ x_hat + B_d @ u_acc, A_d @ P @ A_d.T + Q_ekf

def ekf_update(x_hat, P, z):
    H = np.zeros((3, 9)); H[0:3, 0:3] = np.eye(3)
    y = z - H @ x_hat
    S = H @ P @ H.T + R_kf
    K = P @ H.T @ np.linalg.inv(S)
    return x_hat + K @ y, (np.eye(9) - K @ H) @ P


# =====================================================================
# 3. MUJOCO INNER LOOP CONTROLLERS & VISUALIZER
# =====================================================================
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp; self.kd = kd
    def compute(self, pos_err, vel_err):
        return self.kp * pos_err + self.kd * vel_err

def draw_line(viewer, pt1, pt2, rgba, width=2.0):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: return
    pt1 = np.asarray(pt1, dtype=np.float64).reshape(3,)
    pt2 = np.asarray(pt2, dtype=np.float64).reshape(3,)
    rgba = np.asarray(rgba, dtype=np.float32)
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, float(width), pt1, pt2)
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

nmpc_planner = DroneNMPC(dt=NMPC_DT, N=15)

# Initialization
p_target_true = np.array([1.0, 0.0, 1.0])
v_target_base = np.array([0.15, 0.1, 0.0]) 
# v_target_base = np.array([0, 0, 0.0]) 


x_tar_hat = np.zeros(6); x_tar_hat[0:3] = p_target_true.copy(); x_tar_hat[3:6] = v_target_base.copy()
x_hat = np.zeros(9); x_hat[0:3] = np.array([-2.5, 0.0, 1.5])
u_acc_prev = np.zeros(3)

data.qpos[0:3] = x_hat[0:3]
mujoco.mj_forward(model, data)

last_nmpc_time = -NMPC_DT 
target_roll, target_pitch, target_acc = 0.0, 0.0, GRAVITY
current_phase = 0
last_thrust_state = GRAVITY 
wind_true = np.zeros(3)

# Logs
x_hist, tar_hist, tar_est_hist, u_hist, time_steps_log = [], [], [], [], []
cost_history, delta_history, trust_history = [], [], []
chaser_draw_hist = []

target_hist = []


paused = False
def key_callback(keycode):
    global paused
    if keycode == 32:  # SPACE
        paused = not paused
        print("SIMULATION PAUSED" if paused else "SIMULATION RESUMED")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        
        if not paused:
            # --- 0. WIND DISTURBANCE PHYSICS ---
            wind_true += np.random.normal(0, 0.1, 3) * model.opt.timestep
            wind_true = np.clip(wind_true, -2.5, 2.5) 
            
            # --- 1. MOVE TARGET PHYSICALLY (Base Vel + Wind Drift) ---
            v_target_true = v_target_base + wind_true * 0.5 
            p_target_true += v_target_true * model.opt.timestep
            data.mocap_pos[0] = p_target_true  
            
            # Apply wind to Chaser
            data.qfrc_applied[0:3] = MASS * wind_true

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
                chaser_draw_hist.append(pos.copy()) 
                if len(chaser_draw_hist) > 200: chaser_draw_hist.pop(0) 
                
                
                
                target_hist.append(p_target_true.copy())
                if len(target_hist) > 200:
                    target_hist.pop(0)
                
                # 1. Target EKF (6D)
                z_tar = p_target_true + np.random.normal(0, 0.01, 3)
                x_tar_hat, P_tar = target_predict(x_tar_hat, P_tar)
                x_tar_hat, P_tar = target_update(x_tar_hat, P_tar, z_tar)
                
                # 2. Chaser EKF (9D)
                z_cha = pos + np.random.normal(0, 0.01, 3)
                x_hat, P_ekf = ekf_predict(x_hat, P_ekf, u_acc_prev)
                x_hat, P_ekf = ekf_update(x_hat, P_ekf, z_cha)
                
                # 3. Build 12D State
                state_12d = np.array([
                    x_hat[0], x_hat[1], x_hat[2], 
                    x_hat[3], x_hat[4], x_hat[5], 
                    roll, pitch, last_thrust_state,
                    x_hat[6], x_hat[7], x_hat[8]   # Wind Estimate
                ])
                
                # Phase Logic based on ESTIMATES
                dist_xy = np.linalg.norm(x_hat[0:2] - x_tar_hat[0:2])
                if current_phase == 0 and dist_xy < 0.2:
                    current_phase = 1
                    print(f"[{data.time:.1f}s] FSM TRIGGER: Phase 1 (Cone) Activated!")

                # Solve NMPC using 12D State
                is_first = (last_nmpc_time < 0)
                u_opt, step_cost, step_delta, step_trust = nmpc_planner.solve(state_12d, x_tar_hat, current_phase, is_first_step=is_first)
                
                target_roll, target_pitch, target_acc = u_opt[0], u_opt[1], u_opt[2]
                
                # SAVE LOGS
                x_hist.append(state_12d.copy())
                tar_hist.append(p_target_true.copy())
                tar_est_hist.append(x_tar_hat[0:3].copy())
                u_hist.append(u_opt.copy())
                time_steps_log.append(data.time)
                cost_history.append(step_cost)
                delta_history.append(step_delta)
                trust_history.append(step_trust)

                u_acc_prev = np.array([
                    target_acc * np.sin(target_pitch),
                    -target_acc * np.sin(target_roll) * np.cos(target_pitch),
                    target_acc * np.cos(target_roll) * np.cos(target_pitch) - GRAVITY
                ])
                
                last_thrust_state = target_acc 
                last_nmpc_time = data.time
                print(f"t={data.time:.1f}s | Ph: {current_phase} | Est. Wind: [{x_hat[6]:.2f}, {x_hat[7]:.2f}] m/s²")

            # --- 4. HOLOGRAPHIC VISUALIZATIONS ---
            with viewer.lock():
                viewer.user_scn.ngeom = 0 
                # Past Trajectory 
                for i in range(len(chaser_draw_hist)-1):
                    draw_line(viewer, chaser_draw_hist[i], chaser_draw_hist[i+1], np.array([1, 0, 0, 1]), width=2)
                    
                for i in range(len(target_hist)-1):
                    draw_line(
                        viewer,
                        target_hist[i],
                        target_hist[i+1],
                        np.array([1, 1, 0, 1]),
                        width=3
                    )
                    
                # SCP Lookahead
                lookahead = nmpc_planner.X_nom[:, 0:3]
                for i in range(len(lookahead)-1):
                    draw_line(viewer, lookahead[i], lookahead[i+1], np.array([0, 1, 0, 1]), width=4)

                # Moving Docking Cone
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

# ====================================================================
# IEEE PUBLICATION PLOTS (REVISED & AUTO-SAVING)
# ====================================================================
print("Simulation Ended. Generating IEEE Plots...")
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

# ⭐ Target Estimation Performance
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
ax1.plot([traj[0,0], tar_hist[-1,0]], [traj[0,1], tar_hist[-1,1]], [traj[0,2], tar_hist[-1,2]], 
         color='gray', linestyle=':', linewidth=1.5, label='Nominal Direct Path')
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', linewidth=3.5, label='Executed Chaser Traj')
ax1.plot(tar_hist[:,0], tar_hist[:,1], tar_hist[:,2], 'r-', linewidth=2, label='True Moving Target')
ax1.plot([traj[0,0]], [traj[0,1]], [traj[0,2]], 'go', markersize=6, label='Start')
ax1.plot([tar_hist[-1,0]], [tar_hist[-1,1]], [tar_hist[-1,2]], 'r*', markersize=10, label='Docking Point')

u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.15, edgecolor='none')

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

# ✅ FIG 2: Control Inputs (Split)
fig2, (ax_tilt, ax_thr) = plt.subplots(2, 1, figsize=(4.0, 4.5), sharex=True)
ax_tilt.plot(time_steps, np.degrees(u_hist[:, 0]), 'r-', label='Roll Cmd ($\\phi$)')
ax_tilt.plot(time_steps, np.degrees(u_hist[:, 1]), 'g-', label='Pitch Cmd ($\\theta$)')
ax_tilt.axhline(np.degrees(MAX_TILT), color='k', linestyle='--', linewidth=1.5, label='Tilt Limit')
ax_tilt.axhline(-np.degrees(MAX_TILT), color='k', linestyle='--', linewidth=1.5)
ax_tilt.set_ylabel('Tilt Angle (deg)')
ax_tilt.legend(loc='upper right')

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
ax_v.legend(loc='upper right')
plt.tight_layout()
fig3.savefig("ieee_plots/3_velocity_norm.png", dpi=300, bbox_inches='tight')

# ✅ FIG 4: Obstacle Distance
fig4 = plt.figure(figsize=SINGLE_COL)
ax_obs = plt.gca()
dist_obs = np.linalg.norm(traj - P_OBS, axis=1)
min_dist = np.min(dist_obs)
min_idx = np.argmin(dist_obs)

ax_obs.plot(time_steps, dist_obs, 'r-', linewidth=2)
ax_obs.axhline(R_OBS + R_SAFE, color='k', linestyle='-', linewidth=2.5, label=f'Safe Boundary ({R_OBS + R_SAFE}m)')
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
ax_cone.axhline(np.degrees(THETA), color='r', linestyle='--', linewidth=2, label=f'Violation Limit ({30}°)')
ax_cone.axvline(dock_time, color='k', linestyle=':', linewidth=1.5, label='Docking Achieved')
ax_cone.set_xlabel('Time (s)')
ax_cone.set_ylabel('Approach angle $\\phi$ (deg)')
ax_cone.legend(loc='upper right')
plt.tight_layout()
fig6.savefig("ieee_plots/6_docking_cone_angle.png", dpi=300, bbox_inches='tight')

# ✅ FIG 8: Distance to Target (Interception Performance)
fig8 = plt.figure(figsize=SINGLE_COL)
ax_r = plt.gca()
dist_to_target = np.linalg.norm(traj - tar_hist, axis=1)
ax_r.plot(time_steps, dist_to_target, 'm-', linewidth=2.5, label='Relative Distance')
ax_r.axhline(0.2, color='k', linestyle='--', linewidth=1.5, label='Docking Tolerance (0.2m)')
# ax_r.annotate('Exponential Convergence', xy=(time_steps[len(time_steps)//2], dist_to_target[len(time_steps)//2]), 
#              xytext=(time_steps[len(time_steps)//3], dist_to_target[len(time_steps)//3] + 0.5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
ax_r.set_xlabel('Time (s)')
ax_r.set_ylabel('Distance to Target (m)')
ax_r.legend()
plt.tight_layout()
fig8.savefig("ieee_plots/8_distance_to_target.png", dpi=300, bbox_inches='tight')

plt.show()