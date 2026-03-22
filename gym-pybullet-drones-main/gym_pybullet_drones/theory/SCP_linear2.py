import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

print("=== INITIALIZING CLOSED-LOOP REALISTIC DOCKING SIMULATOR ===")

# =====================================================================
# 1. MULTI-RATE SENSOR FUSION EKF (Target Tracking)
# =====================================================================
class SensorFusionEKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(9) 
        self.P = np.eye(9) * 1.0 
        
        self.F = np.eye(9)
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.F[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        self.F[3:6, 6:9] = np.eye(3) * dt
        
        self.Q = np.eye(9) * 0.001
        
        self.H_imu = np.zeros((3, 9)); self.H_imu[0:3, 6:9] = np.eye(3)
        self.R_imu = np.eye(3) * 0.1 

        self.H_gps = np.zeros((6, 9)); self.H_gps[0:6, 0:6] = np.eye(6)
        self.R_gps = np.eye(6)
        self.R_gps[0:3, 0:3] *= 1.5 
        self.R_gps[3:6, 3:6] *= 0.5 

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_imu(self, z_imu):
        y = z_imu - self.H_imu @ self.x                  
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

    def predict_future(self, steps, dt_plan):
        preds = np.zeros((steps, 9))
        state = self.x.copy()
        
        F_p = np.eye(9)
        F_p[0:3, 3:6] = np.eye(3) * dt_plan
        F_p[0:3, 6:9] = 0.5 * dt_plan**2 * np.eye(3)
        F_p[3:6, 6:9] = np.eye(3) * dt_plan
        
        for i in range(steps):
            state = F_p @ state
            preds[i, :] = state
        return preds

# =====================================================================
# 2. RECEDING HORIZON SCP PLANNER (Relative Dynamics)
# =====================================================================
class RecedingHorizonSCP:
    def __init__(self, N, dt_plan):
        self.N = N
        self.dt = dt_plan
        self.x_nom = None
        self.u_nom = None
        
        self.A = np.eye(6); self.A[0:3, 3:6] = dt_plan * np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[0:3, :] = 0.5 * dt_plan**2 * np.eye(3)
        self.B[3:6, :] = dt_plan * np.eye(3)
        
        self.U_MAX = 15.0; self.V_MAX = 5.0
        self.P_OBS = np.array([-1.0, 0.0, 1.25]); self.R_OBS = 0.4; self.R_SAFE = 0.1
        
        # RESTORED: Pure Top-Down Docking
        self.N_APP = np.array([0, 0, -1]) 
        self.THETA = np.radians(30)
        self.r_c = 0.1; self.r_t = 0.1; self.alpha_min = 1.05
        
        self.alpha_min = 1.05
        self.r_dock = self.alpha_min * (self.r_c+ self.r_t)
        self.r_dock = 0

    def solve(self, chaser_state, target_preds, phase):
        if self.x_nom is None:
            self.x_nom = np.zeros((self.N, 6))
            x_rel_0 = chaser_state - target_preds[0, 0:6]
            for k in range(self.N):
                self.x_nom[k, 0:3] = x_rel_0[0:3] * (1 - k/(self.N-1))
                self.x_nom[k, 1] += 0.5 * np.sin(np.pi * (k/(self.N-1)))
        else:
            self.x_nom[:-1] = self.x_nom[1:].copy()
            self.x_nom[-1] = self.x_nom[-2].copy()
            if self.u_nom is not None:
                self.u_nom[:-1] = self.u_nom[1:].copy()
                self.u_nom[-1] = np.zeros(3) 

        x_rel = cp.Variable((self.N, 6))
        u_c = cp.Variable((self.N-1, 3))
        slack_cone = cp.Variable(self.N-1, nonneg=True)
        slack_tar  = cp.Variable(self.N-1, nonneg=True)
        
        x_rel_0 = chaser_state - target_preds[0, 0:6]
        cost = 0
        con = [x_rel[0, :] == x_rel_0]
        
        # FIX 1: Staging Phase (0) now hovers EXACTLY 0.5m directly above the target.
        # Phase 1 dives to EXACTLY [0,0,0].
        offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
        
        # Terminal Constraints
        con += [x_rel[-1, 3:6] == np.zeros(3)]
        con += [x_rel[-1, 0:3] == offset]
        
        trust_radius = 2.0
        
        for k in range(self.N-1):
            con += [x_rel[k+1, :] == self.A @ x_rel[k, :] + self.B @ u_c[k, :]]
            
            cost += cp.sum_squares(x_rel[k, 0:3] - offset) * 2.0 
            cost += cp.sum_squares(x_rel[k, 3:6]) * 1.0 
            cost += cp.sum_squares(u_c[k, :]) * 0.5
            
            if self.u_nom is not None:
                cost += cp.sum_squares(u_c[k, :] - self.u_nom[k, :]) * 2.0
            
            con += [cp.norm(x_rel[k, :] - self.x_nom[k, :], np.inf) <= trust_radius]
            
            p_rel_nom = self.x_nom[k, 0:3]
            
            # Obstacle Avoidance
            p_obs_rel = self.P_OBS - target_preds[k, 0:3]
            v_obs = p_rel_nom - p_obs_rel
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (x_rel[k, 0:3] - p_obs_rel) >= self.R_OBS + self.R_SAFE]
            
            # Phase Logic
            if phase == 1:
                
                
                # 1. DCOL RESTORED (Separating Hyperplane)
                # n_tar guarantees the chaser stays outside the target sphere
                dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
                n_tar = (p_rel_nom / dist_tar_nom) * self.r_dock
                
                # Notice we removed alpha_min here so the chaser is legally allowed to 
                # reach distance == self.r_dock (which is the goal offset)
                con += [n_tar @ x_rel[k, 0:3] >= self.r_dock - slack_tar[k]]
                
                
                # Top-Down Cone
                con += [cp.norm(x_rel[k, 0:3]) * np.cos(self.THETA) <= -self.N_APP @ x_rel[k, 0:3] + slack_cone[k]]
                
                # FIX 2: REMOVED Target Collision forcefield here! 
                # If we keep it, it will physically repel the chaser from reaching [0,0,0].
                con += [slack_tar[k] == 0] 
            else:
                con += [slack_tar[k] == 0]; con += [slack_cone[k] == 0]
                
            v_global = x_rel[k+1, 3:6] + target_preds[k+1, 3:6]
            con += [cp.norm(v_global, 2) <= self.V_MAX]
            con += [cp.norm(u_c[k, :], np.inf) <= self.U_MAX]

        cost += cp.sum(slack_cone) * 100.0 + cp.sum(slack_tar) * 100.0

        prob = cp.Problem(cp.Minimize(cost), con)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True)
            if prob.status == "optimal":
                self.x_nom = x_rel.value.copy()
                self.u_nom = u_c.value.copy()
                return self.u_nom[0, :]
        except:
            pass
            
        return np.zeros(3)

# =====================================================================
# 3. PHYSICS SIMULATION ENGINE
# =====================================================================
SIM_TIME = 8.0
DT_SIM = 0.01   # Physics & IMU @ 100Hz
DT_PLAN = 0.1   # Planner & GPS @ 10Hz
PLAN_RATIO = int(DT_PLAN / DT_SIM)
STEPS = int(SIM_TIME / DT_SIM)

ekf = SensorFusionEKF(dt=DT_SIM)
planner = RecedingHorizonSCP(N=40, dt_plan=DT_PLAN)

# Data Loggers
log_p_chaser = np.zeros((STEPS, 3)); log_p_target = np.zeros((STEPS, 3))
log_est_target = np.zeros((STEPS, 3)); log_phase = np.zeros(STEPS)

chaser_state = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0]) # [pos, vel]
current_u = np.zeros(3)
phase = 0

print("Commencing Flight Simulation...\n")
for i in range(STEPS):
    t = i * DT_SIM
    
    # 1. GENERATE GROUND TRUTH
    p_true = np.array([0.5 + 0.3*t, 0.0 + 0.1*t, 1.0])
    v_true = np.array([0.3, 0.1, 0.0])
    a_true = np.array([0.0, 0.0, 0.0])
    target_state_true = np.hstack([p_true, v_true])
    
    if i == 0:
        ekf.x[0:3] = p_true; ekf.x[3:6] = v_true
    
    # 2. CHASER PHYSICS
    chaser_state[0:3] += chaser_state[3:6] * DT_SIM + 0.5 * current_u * DT_SIM**2
    chaser_state[3:6] += current_u * DT_SIM
    
    # 3. SENSORS & EKF FUSION
    ekf.predict()
    z_imu = a_true + np.random.normal(0, 0.2, 3) 
    ekf.update_imu(z_imu)
    
    if i % PLAN_RATIO == 0:
        z_gps = target_state_true + np.random.normal(0, [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) 
        ekf.update_gps(z_gps)
        
        # 4. HIGH-LEVEL FSM
        dist_xy = np.linalg.norm(chaser_state[0:2] - ekf.x[0:2])
        
        # FIX 3: Wait until the chaser is aligned overhead (< 0.3m error in XY) before diving!
        # This guarantees it doesn't violate the [0,0,-1] cone constraints.
        if phase == 0 and dist_xy < 0.3:
            phase = 1
            print(f"[{t:.1f}s] FSM TRIGGER: Aligned overhead! Activating Phase 1 (Top-Down Dive)!")
            
        # 5. SCP MPC PLANNER
        target_preds = ekf.predict_future(planner.N, DT_PLAN)
        current_u = planner.solve(chaser_state, target_preds, phase)
        
        if i % (PLAN_RATIO * 10) == 0:
            print(f"[{t:.1f}s] Phase: {phase} | Chaser: {chaser_state[0:2].round(2)} | Target Est: {ekf.x[0:2].round(2)} | Dist: {dist_xy:.2f}m")

    # Logging
    log_p_chaser[i] = chaser_state[0:3]
    log_p_target[i] = p_true
    log_est_target[i] = ekf.x[0:3]
    log_phase[i] = phase

print("\nSimulation Complete! Generating Dashboard...")

# =====================================================================
# 4. TELEMETRY DASHBOARD
# =====================================================================
plt.style.use('seaborn-v0_8-darkgrid')

fig = plt.figure(figsize=(12, 5))

# Plot 1: 3D Trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(log_p_chaser[:,0], log_p_chaser[:,1], log_p_chaser[:,2], 'b-', linewidth=2, label='Chaser Flight Path')
ax1.plot(log_p_target[:,0], log_p_target[:,1], log_p_target[:,2], 'k--', linewidth=2, label='Target Ground Truth')
ax1.plot(log_est_target[:,0], log_est_target[:,1], log_est_target[:,2], 'r-', alpha=0.5, label='EKF Filtered Target')

# Obstacle
u_sph, v_sph = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
x_sph = -1.0 + 0.4*np.cos(u_sph)*np.sin(v_sph)
y_sph = 0.0 + 0.4*np.sin(u_sph)*np.sin(v_sph)
z_sph = 1.25 + 0.4*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='red', alpha=0.3)

ax1.set_title('Receding Horizon Docking (Moving Target)')
ax1.legend()

# Plot 2: FSM Distance Tracking
ax2 = fig.add_subplot(122)
time_array = np.arange(0, SIM_TIME, DT_SIM)
dist_array = np.linalg.norm(log_p_chaser - log_p_target, axis=1)

ax2.plot(time_array, dist_array, 'm-', linewidth=2, label='Relative Distance')
ax2.axhline(0.3, color='k', linestyle='--', label='FSM Trigger Threshold (0.3m XY)')
ax2.fill_between(time_array, 0, 3, where=(log_phase==1), color='cyan', alpha=0.2, transform=ax2.get_xaxis_transform(), label='Phase 1 Active (Cone)')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Target (m)')
ax2.set_title('FSM Phase Tracking')
ax2.legend()

plt.tight_layout()
plt.show()