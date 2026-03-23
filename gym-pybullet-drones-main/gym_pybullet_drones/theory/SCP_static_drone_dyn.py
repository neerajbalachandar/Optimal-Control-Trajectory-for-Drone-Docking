import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ================= SYSTEM =================
dt = 0.1
N = 25  # Planning horizon

U_MAX = 15.0 # Max Thrust (m/s^2)
U_MIN = 2.0  # Min Thrust (m/s^2)
MAX_TILT = np.radians(25) # Max Roll/Pitch
V_MAX = 5.0

P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)
N_APP = np.array([0, 0, -1])

r_dock = 0 
GRAVITY = 9.81

# 9D State: [px, py, pz, vx, vy, vz, phi(roll), theta(pitch), a_T(thrust)]
x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, GRAVITY])
p_target_true = np.array([0.5, 0.0, 1.0]) 
p_target = p_target_true.copy()

# ================= FULL NON-LINEAR DYNAMICS =================
def f_dyn(x, u):
    """ True Continuous Non-Linear Quadrotor Dynamics """
    px, py, pz, vx, vy, vz, phi, theta, a_T = x
    phi_cmd, theta_cmd, a_cmd = u
    
    tau_rp = 0.1  # Rotor Roll/Pitch Time Constant
    tau_t = 0.05  # Motor Spool-Up Time Constant

    return np.array([
        vx, 
        vy, 
        vz,
        a_T * np.sin(theta),                                # dvx
        -a_T * np.sin(phi) * np.cos(theta),                 # dvy
        a_T * np.cos(phi) * np.cos(theta) - GRAVITY,        # dvz
        (phi_cmd - phi) / tau_rp,                           # dphi
        (theta_cmd - theta) / tau_rp,                       # dtheta
        (a_cmd - a_T) / tau_t                               # dThrust
    ])

def get_jacobians(x, u):
    """ Calculate Analytic Jacobians for Successive Convexification """
    px, py, pz, vx, vy, vz, phi, theta, a_T = x
    tau_rp = 0.1
    tau_t = 0.05

    Ac = np.zeros((9, 9))
    Ac[0, 3] = Ac[1, 4] = Ac[2, 5] = 1.0

    Ac[3, 7] = a_T * np.cos(theta)
    Ac[3, 8] = np.sin(theta)

    Ac[4, 6] = -a_T * np.cos(phi) * np.cos(theta)
    Ac[4, 7] = a_T * np.sin(phi) * np.sin(theta)
    Ac[4, 8] = -np.sin(phi) * np.cos(theta)

    Ac[5, 6] = -a_T * np.sin(phi) * np.cos(theta)
    Ac[5, 7] = -a_T * np.cos(phi) * np.sin(theta)
    Ac[5, 8] = np.cos(phi) * np.cos(theta)

    Ac[6, 6] = -1.0 / tau_rp
    Ac[7, 7] = -1.0 / tau_rp
    Ac[8, 8] = -1.0 / tau_t

    Bc = np.zeros((9, 3))
    Bc[6, 0] = 1.0 / tau_rp
    Bc[7, 1] = 1.0 / tau_rp
    Bc[8, 2] = 1.0 / tau_t

    return Ac, Bc

# ================= NMPC INITIALIZATION =================
X_nom = np.zeros((N, 9))
for k in range(N):
    al = k / (N - 1)
    X_nom[k, 0:3] = x0[0:3] + al * (p_target_true - x0[0:3])
    X_nom[k, 1] += 0.5 * np.sin(np.pi * al) 
    X_nom[k, 8] = GRAVITY 

u_nom = np.zeros((N-1, 3))
u_nom[:, 2] = GRAVITY

SIM_MAX_STEPS = 80
TOL = 1e-3
MAX_ITERS = 10 

x_hist = [x0.copy()]
u_hist = []
cost_history = []
phase_hist = []
time_steps_hist = [0.0]

x_true = x0.copy()
phase = 0 

print("Starting Full Non-Linear MPC (NMPC) Simulation...")

# ================= ONLINE NMPC LOOP =================
for sim_step in range(SIM_MAX_STEPS):
    
    # 1. Measurement & Target
    sensor_noise = np.random.normal(0, 0.02, 3) 
    p_target = p_target_true + sensor_noise
    
    dist_to_goal = np.linalg.norm(x_true[0:3] - p_target_true)
    vel_mag = np.linalg.norm(x_true[3:6])
    if dist_to_goal < 0.15 and vel_mag < 0.2:
        print(f"Goal Reached at step {sim_step}!")
        break

    # 2. Phase Logic
    dist_xy = np.linalg.norm(x_true[0:2] - p_target[0:2])
    if phase == 0 and dist_xy < 0.3:
        phase = 1
        print(f"[{sim_step*dt:.1f}s] FSM TRIGGER: Phase 1 (Cone) Activated!")

    # 3. Warm Start NMPC
    if sim_step > 0:
        X_nom[:-1, :] = X_nom[1:, :]
        X_nom[-1, :] = X_nom[-2, :]
        u_nom[:-1, :] = u_nom[1:, :]
        u_nom[-1, :] = np.array([0.0, 0.0, GRAVITY])
        
    trust_radius = 2.0
    scp_converged = False
    
    # 4. Successive Convexification Loop
   # 4. Successive Convexification Loop
    for it in range(MAX_ITERS):
        X = cp.Variable((N, 9))
        u = cp.Variable((N-1, 3))
        
        slack_cone = cp.Variable(N-1, nonneg=True)
        slack_tar  = cp.Variable(N-1, nonneg=True)
        
        # --- VIRTUAL CONTROL SLACKS (The magic fix for 'inf' crashes) ---
        nu = cp.Variable((N-1, 9)) 
        
        cost = 0
        con = [X[0, :] == x_true]
        
        offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
        
        # --- SOFT TERMINAL CONSTRAINTS ---
        # Instead of strict equalities, we heavily penalize missing the target
        cost += cp.sum_squares(X[-1, 0:3] - (p_target + offset)) * 1000.0
        cost += cp.sum_squares(X[-1, 3:6]) * 500.0
        cost += cp.sum_squares(X[-1, 6:8]) * 500.0
        
        for k in range(N-1):
            x_k_nom = X_nom[k, :]
            u_k_nom = u_nom[k, :]
            
            f_k = f_dyn(x_k_nom, u_k_nom)
            Ac, Bc = get_jacobians(x_k_nom, u_k_nom)
            
            # Dynamics with Virtual Slack 'nu'
            con += [X[k+1, :] == X[k, :] + dt * (f_k + Ac @ (X[k, :] - x_k_nom) + Bc @ (u[k, :] - u_k_nom)) + nu[k, :]]
            
            # Massive penalty on 'nu' so the solver only uses it to avoid an 'inf' crash
            cost += cp.sum_squares(nu[k, :]) * 1e5
            
            # --- COST FUNCTION ---
            p_rel = X[k, 0:3] - p_target
            cost += cp.sum_squares(p_rel - offset) * 2.0 
            cost += cp.sum_squares(X[k, 3:6]) * 1.0 
            cost += cp.sum_squares(X[k, 6:8]) * 5.0 # Penalize aggressive tilt
            cost += cp.sum_squares(u[k, 0:2]) * 2.0 # Penalize erratic commands
            cost += cp.sum_squares(u[k, 2] - GRAVITY) * 0.1 
            cost += cp.sum_squares(u[k, :] - u_k_nom) * 2.0 # SCP Regularization
            
            # --- CONSTRAINTS ---
            if k > 0: # Don't apply trust region to the initial shifted state!
                con += [cp.norm(X[k, :] - x_k_nom, np.inf) <= trust_radius]
                
            con += [cp.norm(u[k, :] - u_k_nom, np.inf) <= trust_radius]
            
            con += [u[k, 0] >= -MAX_TILT, u[k, 0] <= MAX_TILT] 
            con += [u[k, 1] >= -MAX_TILT, u[k, 1] <= MAX_TILT] 
            con += [u[k, 2] >= U_MIN, u[k, 2] <= U_MAX]        
            con += [cp.norm(X[k+1, 3:6], 2) <= V_MAX]
            
            # Obstacle Lin
            p_rel_nom = X_nom[k, 0:3]
            p_obs_rel = P_OBS - p_target
            v_obs = p_rel_nom - P_OBS
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (X[k, 0:3] - P_OBS) >= R_OBS + R_SAFE]

            if phase == 1:
                dist_tar_nom = np.linalg.norm(p_rel_nom - p_target) + 1e-8
                n_tar = ((p_rel_nom - p_target) / dist_tar_nom) * r_dock
                con += [n_tar @ p_rel >= r_dock - slack_tar[k]]
                con += [cp.norm(p_rel) * np.cos(THETA) <= -N_APP @ p_rel + slack_cone[k]]
                con += [slack_tar[k] == 0] 
            else:
                con += [slack_tar[k] == 0]
                con += [slack_cone[k] == 0]

        con += [cp.norm(X[-1, :] - X_nom[-1, :], np.inf) <= trust_radius]
        cost += cp.sum(slack_cone) * 100.0 
        cost += cp.sum(slack_tar) * 100.0

        prob = cp.Problem(cp.Minimize(cost), con)
        
        # Use CLARABEL. ECOS handles non-linear slack variables very poorly.
        prob.solve(solver=cp.CLARABEL, warm_start=True, ignore_dpp=True) 
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            trust_radius *= 0.5
            continue
            
        delta = np.linalg.norm(X.value - X_nom, np.inf)
        X_nom = X.value.copy()
        u_nom = u.value.copy() 
        
        if delta < TOL:
            scp_converged = True
            break
        trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

    print(f"Step {sim_step:02d} | Phase: {phase} | Cost: {prob.value:.1f} | Iters: {it+1}")

    # 5. Integrate True Non-Linear Dynamics
    u_opt = u_nom[0, :]
    
    # We use micro-stepping for accurate continuous simulation of the drone body
    dt_sim = 0.01
    for _ in range(int(dt/dt_sim)):
        x_true = x_true + dt_sim * f_dyn(x_true, u_opt)
        
    x_hist.append(x_true.copy())
    u_hist.append(u_opt.copy())
    phase_hist.append(phase)
    cost_history.append(prob.value if prob.value else 0)
    time_steps_hist.append((sim_step + 1) * dt)

# ================= PLOTTING (Exact match to SCP_static_EKF.py) =================
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
phase_hist = np.array(phase_hist)
time_steps_hist = np.array(time_steps_hist)

plt.style.use('seaborn-v0_8-darkgrid')

# ----------------- FIGURE 1: 3D Trajectory & FSM -----------------
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(121, projection='3d')

traj = x_hist[:, 0:3]
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b.-', linewidth=3, label='NMPC Executed Traj')
ax1.plot(x0[0], x0[1], x0[2], 'go', markersize=8, label='Start')
ax1.plot(p_target_true[0], p_target_true[1], p_target_true[2], 'r*', markersize=12, label='True Target')

u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3)

ax1.set_title('Online NMPC Trajectory (9D Drone Dynamics)')
ax1.legend()

ax2 = fig1.add_subplot(122)
ctrl_steps_hist = time_steps_hist[:-1]
dist_array = np.linalg.norm(x_hist[:-1, 0:3] - p_target_true, axis=1)

ax2.plot(ctrl_steps_hist, dist_array, 'm-', linewidth=2, label='Relative Distance')
ax2.axhline(0.3, color='k', linestyle='--', label='FSM Trigger Threshold')
ax2.fill_between(ctrl_steps_hist, 0, max(dist_array), where=(phase_hist==1), color='cyan', alpha=0.2, transform=ax2.get_xaxis_transform(), label='Phase 1 Active (Cone)')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Target (m)')
ax2.set_title('FSM Phase Tracking')
ax2.legend()

# ----------------- FIGURE 2: Safety Constraints -----------------
fig3, (ax_obs, ax_cone) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

dist_obs = np.linalg.norm(traj - P_OBS, axis=1)
ax_obs.plot(time_steps_hist, dist_obs, 'r', linewidth=2)
ax_obs.axhline(R_OBS + R_SAFE, color='k', linestyle='--', label=f'Minimum Safe Distance ({R_OBS + R_SAFE}m)')
ax_obs.fill_between(time_steps_hist, 0, R_OBS + R_SAFE, color='red', alpha=0.1)
ax_obs.set_ylabel('Distance to Obstacle Center (m)')
ax_obs.set_title('Obstacle Avoidance Clearance')
ax_obs.legend()

angles = []
for p in x_hist[:, 0:3]:
    p_rel = p - p_target_true
    dist = np.linalg.norm(p_rel)
    if dist < 1e-5:
        angles.append(0.0)
    else:
        cos_phi = np.clip(np.dot(-N_APP, p_rel) / dist, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_phi)))

ax_cone.plot(time_steps_hist, angles, 'c', linewidth=2)
ax_cone.axhline(np.degrees(THETA), color='k', linestyle='--', label=f'Cone Limit ({np.degrees(THETA)}°)')
ax_cone.fill_between(time_steps_hist, 0, np.degrees(THETA), color='cyan', alpha=0.1)
ax_cone.set_ylabel('Approach Angle (deg)')
ax_cone.set_xlabel('Time (s)')
ax_cone.set_title('Executed Docking Cone Alignment')
ax_cone.legend()

plt.tight_layout()

# Save the trajectory for MuJoCo to track blindly
np.save("x_ref.npy", x_hist)
np.save("u_ref.npy", u_hist)
print("Trajectory saved to x_ref.npy and u_ref.npy! Ready for MuJoCo.")


plt.show()