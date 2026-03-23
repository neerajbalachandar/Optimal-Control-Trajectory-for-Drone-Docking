import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

SCP_SOLVER_ORDER = ("ECOS", "CLARABEL", "SCS")
ACCEPTABLE_SOLVER_STATUSES = {"optimal", "optimal_inaccurate"}


def solve_with_fallback(prob: cp.Problem, warm_start: bool = True) -> str:
    """Solve SCP subproblem with robust conic solver fallback."""
    installed = set(cp.installed_solvers())
    attempted = []
    last_status = None

    for solver_name in SCP_SOLVER_ORDER:
        if solver_name not in installed:
            continue
        attempted.append(solver_name)

        solver_kwargs = {}
        if solver_name == "SCS":
            solver_kwargs["max_iters"] = 6000
            solver_kwargs["eps"] = 1e-4

        try:
            prob.solve(solver=solver_name, warm_start=warm_start, **solver_kwargs)
        except cp.error.SolverError:
            continue

        last_status = prob.status
        if prob.status in ACCEPTABLE_SOLVER_STATUSES:
            return solver_name

    if not attempted:
        raise cp.error.SolverError(
            "No supported conic solver found. Install at least one of: ECOS, CLARABEL, SCS."
        )
    raise cp.error.SolverError(
        f"All SCP solvers failed or returned non-optimal status. Tried {attempted}, last status={last_status}."
    )

# =====================================================================
# 1. MULTI-RATE SENSOR FUSION EKF (Target Tracking)
# =====================================================================
class SensorFusionEKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(9) # [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]
        self.P = np.eye(9) * 1.0 
        
        # State Transition Matrix
        self.F = np.eye(9)
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.F[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        self.F[3:6, 6:9] = np.eye(3) * dt
        
        self.Q = np.eye(9) * 0.001
        
        # Measurement Models
        self.H_imu = np.zeros((3, 9)); self.H_imu[0:3, 6:9] = np.eye(3)
        self.R_imu = np.eye(3) * 0.05 

        self.H_gps = np.zeros((6, 9)); self.H_gps[0:6, 0:6] = np.eye(6)
        self.R_gps = np.eye(6) * 0.02
        self.R_gps[0:3, 0:3] *= 1.5 # GPS position slightly noisier

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_imu(self, z_acc):
        y = z_acc - self.H_imu @ self.x
        S = self.H_imu @ self.P @ self.H_imu.T + self.R_imu
        K = self.P @ self.H_imu.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H_imu) @ self.P

    def update_gps(self, z_gps):
        y = z_gps - self.H_gps @ self.x
        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps
        K = self.P @ self.H_gps.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H_gps) @ self.P

# ================= SYSTEM CONFIG =================
dt = 0.1
N = 25  # Planning horizon

U_MAX = 15.0
V_MAX = 5.0

# Absolute obstacle position
P_OBS = np.array([-0.5, 0.3, 1.25]) 
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)
N_APP = np.array([0, 0, -1])

r_c = 0.1
r_t = 0.1
alpha_min = 1.05
r_dock = 0.0 

# INITIAL STATES
x_chaser_true = np.array([-2.5, -0.5, 1.5, 0.0, 0.0, 0.0])

# Target moving linearly
x_tar_true = np.array([0.0, 0.0, 1.0,  0.2, 0.1, 0.0]) 

# ================= DYNAMICS =================
A_d = np.eye(6)
A_d[0:3, 3:6] = dt * np.eye(3)

B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
B_d[3:6, :] = dt * np.eye(3)

# Initialize EKF
ekf = SensorFusionEKF(dt)
ekf.x[0:6] = x_tar_true.copy() # Warm start EKF

# ================= INITIAL GUESS (Relative) =================
x_nom = np.zeros((N, 6))
x_rel_0 = x_chaser_true - x_tar_true

for k in range(N):
    al = k / (N - 1)
    x_nom[k, 0:3] = (1 - al) * x_rel_0[0:3] 
    x_nom[k, 1] += 0.5 * np.sin(np.pi * al) 

# ================= ONLINE PLANNING LOOP =================
SIM_MAX_STEPS = 90
TOL = 1e-3
MAX_ITERS = 10 

# Logging
x_hist, u_hist, phase_hist, tar_hist, tar_est_hist = [], [], [], [], []
x_nom_hist = []

u_nom = np.zeros((N-1, 3))
phase = 0 # 0: Hover Above, 1: Dive

print("Starting Closed-Loop Online SCP (Moving Target + EKF)...")

for sim_step in range(SIM_MAX_STEPS):
    
    # ---------------------------------------------------------
    # 1. SENSOR FUSION (EKF) & TARGET LOOKAHEAD
    # ---------------------------------------------------------
    # Simulate target moving (Constant Velocity)
    x_tar_true = A_d @ x_tar_true
    
    # Generate noisy sensor readings
    z_gps = x_tar_true + np.random.normal(0, [0.03, 0.03, 0.03, 0.01, 0.01, 0.01])
    z_imu = np.zeros(3) + np.random.normal(0, 0.05, 3) # True accel is 0
    
    # EKF Cycle
    ekf.predict()
    ekf.update_imu(z_imu)
    if sim_step % 2 == 0: # GPS updates slower (every 2nd step)
        ekf.update_gps(z_gps)
        
    x_tar_est = ekf.x[0:6].copy()
    
    # Predict target's absolute state over the horizon N
    x_tar_pred = np.zeros((N, 6))
    x_tar_pred[0] = x_tar_est
    for k in range(1, N):
        x_tar_pred[k] = A_d @ x_tar_pred[k-1]

    # ---------------------------------------------------------
    # 2. STATE CALCULATION & FSM
    # ---------------------------------------------------------
    # Current relative state (using EKF estimate!)
    x_rel_true = x_chaser_true - x_tar_est
    
    # Check stopping condition (docked and velocities matched)
    dist_to_goal = np.linalg.norm(x_rel_true[0:3])
    vel_error = np.linalg.norm(x_rel_true[3:6])
    if dist_to_goal < 0.1 and vel_error < 0.15 and phase == 1:
        print(f"Soft Docking Achieved at step {sim_step}!")
        break

    # FSM Phase Logic
    dist_xy = np.linalg.norm(x_rel_true[0:2])
    if phase == 0 and dist_xy < 0.3:
        phase = 1
        print(f"[{sim_step*dt:.1f}s] FSM TRIGGER: Target acquired overhead! Activating Phase 1 (Dive)!")

    # Warm-start
    if sim_step > 0:
        x_nom[:-1, :] = x_nom[1:, :]
        x_nom[-1, :] = x_nom[-2, :]
        u_nom[:-1, :] = u_nom[1:, :]
        u_nom[-1, :] = np.zeros(3)
        
    trust_radius = 2.0
    scp_converged = False
    last_prob_value = np.nan
    last_solver = "none"
    
    # ---------------------------------------------------------
    # 3. SCP OPTIMIZATION (RELATIVE FRAME)
    # ---------------------------------------------------------
    for it in range(MAX_ITERS):
        x_rel = cp.Variable((N, 6))
        u = cp.Variable((N-1, 3))
        slack_cone = cp.Variable(N-1, nonneg=True)
        slack_tar  = cp.Variable(N-1, nonneg=True)
        
        cost = 0
        con = [x_rel[0, :] == x_rel_true] 
        
        # Offset is [0,0,0.5] to hover above, or [0,0,0] to dock
        offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
        
        # Terminal constraint forces relative velocity to 0 (matches target velocity!)
        con += [x_rel[-1, 0:3] == offset] 
        con += [x_rel[-1, 3:6] == np.zeros(3)]
        
        for k in range(N-1):
            con += [x_rel[k+1, :] == A_d @ x_rel[k, :] + B_d @ u[k, :]]
            
            # Penalize deviation from offset, and penalize relative velocity
            cost += cp.sum_squares(x_rel[k, 0:3] - offset) * 2.0 
            cost += cp.sum_squares(x_rel[k, 3:6]) * 1.5 
            cost += cp.sum_squares(u[k, :]) * 0.5
            cost += cp.sum_squares(u[k, :] - u_nom[k, :]) * 1.0
            
            con += [cp.norm(x_rel[k, :] - x_nom[k, :], np.inf) <= trust_radius]
            
            # ---------------------------------------------------------
            # LOOKAHEAD OBSTACLE AVOIDANCE
            # Because target is moving, the static obstacle "moves backwards" 
            # in the relative frame. We must use the predicted target position!
            # ---------------------------------------------------------
            p_rel_nom = x_nom[k, 0:3]
            p_obs_rel = P_OBS - x_tar_pred[k, 0:3] # Dynamic Relative Obstacle
            
            v_obs = p_rel_nom - p_obs_rel
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (x_rel[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

            # ---------------------------------------------------------
            # DOCKING CONE (Target is always at [0,0,0] in relative frame)
            # ---------------------------------------------------------
            if phase == 1:
                dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
                n_tar = (p_rel_nom / dist_tar_nom) * r_dock
                
                con += [n_tar @ x_rel[k, 0:3] >= r_dock - slack_tar[k]]
                con += [cp.norm(x_rel[k, 0:3]) * np.cos(THETA) <= -N_APP @ x_rel[k, 0:3] + slack_cone[k]]
                con += [slack_tar[k] == 0] 
            else:
                con += [slack_tar[k] == 0]
                con += [slack_cone[k] == 0]
                
            con += [cp.norm(u[k, :], np.inf) <= U_MAX]
            con += [cp.norm(x_rel[k+1, 3:6], 2) <= V_MAX]

        con += [cp.norm(x_rel[-1, :] - x_nom[-1, :], np.inf) <= trust_radius]
        cost += cp.sum(slack_cone) * 100.0 
        cost += cp.sum(slack_tar) * 100.0

        prob = cp.Problem(cp.Minimize(cost), con)
        try:
            last_solver = solve_with_fallback(prob, warm_start=True)
        except cp.error.SolverError:
            trust_radius *= 0.5
            continue

        if x_rel.value is None or u.value is None:
            trust_radius *= 0.5
            continue

        last_prob_value = float(prob.value) if prob.value is not None else np.nan
        delta = np.linalg.norm(x_rel.value - x_nom, np.inf)
        x_nom = x_rel.value.copy()
        u_nom = u.value.copy() 
        
        if delta < TOL:
            scp_converged = True
            break
        trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

    cost_str = f"{last_prob_value:.1f}" if np.isfinite(last_prob_value) else "nan"
    print(
        f"Step {sim_step:02d} | Cost: {cost_str} | Dist: {dist_to_goal:.2f} | "
        f"Vel Err: {vel_error:.2f} | Solver: {last_solver} | SCP Converged: {scp_converged}"
    )

    # 4. Apply Control to TRUE CHASER dynamics (Absolute frame)
    u_cmd = u_nom[0, :]
    x_chaser_true = A_d @ x_chaser_true + B_d @ u_cmd
    
    # Save History
    x_hist.append(x_chaser_true.copy())
    u_hist.append(u_cmd.copy())
    phase_hist.append(phase)
    tar_hist.append(x_tar_true.copy())
    tar_est_hist.append(x_tar_est.copy())

# ====================================================================
# ========================= PLOTTING DASHBOARDS ======================
# ====================================================================
x_hist = np.array(x_hist)
tar_hist = np.array(tar_hist)
tar_est_hist = np.array(tar_est_hist)
phase_hist = np.array(phase_hist)
ctrl_steps = np.arange(len(u_hist)) * dt

plt.style.use('seaborn-v0_8-darkgrid')
fig1 = plt.figure(figsize=(14, 6))

# ----------------- FIGURE 1: 3D Trajectory -----------------
ax1 = fig1.add_subplot(121, projection='3d')
ax1.plot(x_hist[:,0], x_hist[:,1], x_hist[:,2], 'b.-', linewidth=2, label='Chaser Trajectory')
ax1.plot(tar_hist[:,0], tar_hist[:,1], tar_hist[:,2], 'r-', linewidth=2, label='True Moving Target')
ax1.plot(tar_est_hist[:,0], tar_est_hist[:,1], tar_est_hist[:,2], 'y--', alpha=0.7, label='EKF Target Estimate')

# Obstacle Sphere
u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3)

ax1.set_title('Online Closed-Loop Tracking (Linearly Moving Target)')
ax1.legend()

# ----------------- FIGURE 2: Relative Metrics -----------------
ax2 = fig1.add_subplot(122)

# Distance
rel_dist = np.linalg.norm(x_hist[:, 0:3] - tar_hist[:, 0:3], axis=1)
ax2.plot(ctrl_steps, rel_dist, 'm-', linewidth=2, label='Relative Position Error (m)')

# Velocity Matching Error
rel_vel = np.linalg.norm(x_hist[:, 3:6] - tar_hist[:, 3:6], axis=1)
ax2.plot(ctrl_steps, rel_vel, 'c-', linewidth=2, label='Relative Velocity Error (m/s)')

ax2.axhline(0.3, color='k', linestyle='--', alpha=0.5, label='FSM Activation Radius')
ax2.fill_between(ctrl_steps, 0, max(rel_dist), where=(phase_hist==1), color='cyan', alpha=0.1, transform=ax2.get_xaxis_transform(), label='Phase 1 Active (Dive)')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error Magnitude')
ax2.set_title('Soft-Docking Convergence Metrics')
ax2.legend()

plt.tight_layout()
plt.show()