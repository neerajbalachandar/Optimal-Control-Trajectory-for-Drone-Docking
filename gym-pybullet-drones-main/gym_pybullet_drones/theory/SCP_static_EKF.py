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

# ================= SYSTEM =================
dt = 0.1
N = 25  # Planning horizon

U_MAX = 15.0
V_MAX = 5.0

P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)
N_APP = np.array([0, 0, -1])

r_c = 0.1
r_t = 0.1
alpha_min = 1.05
r_dock = 0 # From Code 2: allows reaching exactly [0,0,0]

x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
p_target_true = np.array([0.5, 0.0, 1.0]) # The actual static target

# Global p_target used by the solver (will be updated with noise online)
p_target = p_target_true.copy()

# ================= DYNAMICS =================
A_d = np.eye(6)
A_d[0:3, 3:6] = dt * np.eye(3)

B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
B_d[3:6, :] = dt * np.eye(3)

# ================= INITIAL GUESS (Relative Frame) =================
# Since we optimize in relative frame, x_nom must be relative
x_nom = np.zeros((N, 6))
x_tar_full = np.hstack([p_target, np.zeros(3)])
x_rel_0 = x0 - x_tar_full

for k in range(N):
    al = k / (N - 1)
    x_nom[k, 0:3] = (1 - al) * x_rel_0[0:3] # Drives to [0,0,0]
    # Bow laterally to avoid obstacle initially
    x_nom[k, 1] += 0.5 * np.sin(np.pi * al) 
    
# ================= ONLINE PLANNING LOOP (MPC) =================
SIM_MAX_STEPS = 80
TOL = 1e-3
MAX_ITERS = 10 

# History arrays for the ACTUAL executed path
x_hist = [x0.copy()]
u_hist = []
cost_history = []
delta_history = []
trust_history = []
phase_hist = []

x_true = x0.copy()
u_nom = np.zeros((N-1, 3))
phase = 0 # FSM Phase Tracker

print("Starting Closed-Loop Online SCP (Relative Frame)...")

for sim_step in range(SIM_MAX_STEPS):
    # 1. Sensor Fusion: Predict target with slight static noise
    sensor_noise = np.random.normal(0, 0.02, 3) # 2cm standard deviation
    p_target = p_target_true + sensor_noise
    x_tar_full = np.hstack([p_target, np.zeros(3)])
    
    # Check if we reached the target
    dist_to_goal = np.linalg.norm(x_true[0:3] - p_target_true)
    vel_mag = np.linalg.norm(x_true[3:6])
    if dist_to_goal < 0.15 and vel_mag < 0.2:
        print(f"Goal Reached at step {sim_step}!")
        break
        
    # Calculate True Relative State
    x_rel_true = x_true - x_tar_full

    # 2. FSM Phase Logic (Wait until overhead before diving)
    dist_xy = np.linalg.norm(x_rel_true[0:2])
    if phase == 0 and dist_xy < 0.3:
        phase = 1
        print(f"[{sim_step*dt:.1f}s] FSM TRIGGER: Aligned overhead! Activating Phase 1 (Top-Down Dive)!")

    # 3. Warm-start the initial guess
    if sim_step > 0:
        x_nom[:-1, :] = x_nom[1:, :]
        x_nom[-1, :] = x_nom[-2, :]
        u_nom[:-1, :] = u_nom[1:, :]
        u_nom[-1, :] = np.zeros(3)
        
    trust_radius = 2.0
    scp_converged = False
    last_prob_value = np.nan
    last_delta = np.nan
    last_solver = "none"
    
    # 4. Inner SCP Optimization Loop (RELATIVE FRAME)
    for it in range(MAX_ITERS):
        x_rel = cp.Variable((N, 6))
        u = cp.Variable((N-1, 3))
        
        slack_cone = cp.Variable(N-1, nonneg=True)
        slack_tar  = cp.Variable(N-1, nonneg=True)
        
        cost = 0
        con = [x_rel[0, :] == x_rel_true] # Start at current RELATIVE state
        
        # Staging Phase (0) hovers 0.5m above. Phase 1 dives to [0,0,0].
        offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
        
        # Terminal Boundary Condition
        con += [x_rel[-1, 0:3] == offset] 
        con += [x_rel[-1, 3:6] == np.zeros(3)]
        
        for k in range(N-1):
            # Since target velocity is 0, relative dynamics = absolute dynamics
            con += [x_rel[k+1, :] == A_d @ x_rel[k, :] + B_d @ u[k, :]]
            
            # Code 2 Cost Formulation
            cost += cp.sum_squares(x_rel[k, 0:3] - offset) * 2.0 
            cost += cp.sum_squares(x_rel[k, 3:6]) * 1.0 
            cost += cp.sum_squares(u[k, :]) * 0.5
            
            # u_nom smoothing
            cost += cp.sum_squares(u[k, :] - u_nom[k, :]) * 2.0
            
            # Trust Region
            con += [cp.norm(x_rel[k, :] - x_nom[k, :], np.inf) <= trust_radius]
            
            p_rel_nom = x_nom[k, 0:3]
            
            # Obstacle Avoidance (Relative Frame)
            p_obs_rel = P_OBS - p_target
            v_obs = p_rel_nom - p_obs_rel
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (x_rel[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

            # Phase Logic
            if phase == 1:
                # DCOL Separating Hyperplane
                dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
                n_tar = (p_rel_nom / dist_tar_nom) * r_dock
                
                con += [n_tar @ x_rel[k, 0:3] >= r_dock - slack_tar[k]]
                
                # Top-Down Cone
                con += [cp.norm(x_rel[k, 0:3]) * np.cos(THETA) <= -N_APP @ x_rel[k, 0:3] + slack_cone[k]]
                con += [slack_tar[k] == 0] 
            else:
                con += [slack_tar[k] == 0]
                con += [slack_cone[k] == 0]
                
            # Physical Limits (Relative velocity = Absolute velocity since target is static)
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
        last_delta = float(delta)
        
        x_nom = x_rel.value.copy()
        u_nom = u.value.copy() 
        
        if delta < TOL:
            scp_converged = True
            break
            
        trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

    # Save metrics
    cost_history.append(last_prob_value)
    delta_history.append(last_delta)
    trust_history.append(trust_radius)

    cost_str = f"{last_prob_value:.1f}" if np.isfinite(last_prob_value) else "nan"
    print(
        f"Sim Step {sim_step:02d} | Phase: {phase} | Cost: {cost_str} | "
        f"dist_to_goal: {dist_to_goal:.2f} | Solver: {last_solver} | SCP Converged: {scp_converged}"
    )

    # 4. Apply Control
    u_cmd = u_nom[0, :]
    
    # 5. Propagate True Dynamics (Absolute Frame)
    x_true = A_d @ x_true + B_d @ u_cmd
    
    # 6. Save History
    x_hist.append(x_true.copy())
    u_hist.append(u_cmd.copy())
    phase_hist.append(phase)

# Convert history to numpy arrays for plotting
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
phase_hist = np.array(phase_hist)

# Dynamically generate time arrays
actual_steps = len(u_hist)
time_steps_hist = np.arange(actual_steps + 1) * dt
ctrl_steps_hist = np.arange(actual_steps) * dt

# ====================================================================
# ========================= PLOTTING DASHBOARDS ======================
# ====================================================================

plt.style.use('seaborn-v0_8-darkgrid')

# ----------------- FIGURE 1: 3D Trajectory -----------------
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(121, projection='3d')
traj = x_hist[:, 0:3]
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b.-', linewidth=3, label='Executed Chaser Traj')
ax1.plot(x0[0], x0[1], x0[2], 'go', markersize=8, label='Start')
ax1.plot(p_target_true[0], p_target_true[1], p_target_true[2], 'r*', markersize=12, label='True Target')

# Obstacle Sphere
u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3)
ax1.set_title('Online Closed-Loop Trajectory (Relative Plan)')
ax1.legend()

# ----------------- FIGURE 1b: FSM Tracking -----------------
ax2 = fig1.add_subplot(122)
dist_array = np.linalg.norm(x_hist[:-1, 0:3] - p_target_true, axis=1)

ax2.plot(ctrl_steps_hist, dist_array, 'm-', linewidth=2, label='Relative Distance')
ax2.axhline(0.3, color='k', linestyle='--', label='FSM Trigger Threshold (0.3m XY)')
ax2.fill_between(ctrl_steps_hist, 0, max(dist_array), where=(phase_hist==1), color='cyan', alpha=0.2, transform=ax2.get_xaxis_transform(), label='Phase 1 Active (Cone)')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Target (m)')
ax2.set_title('FSM Phase Tracking')
ax2.legend()


# ----------------- FIGURE 2: Safety Constraints -----------------
fig3, (ax_obs, ax_cone) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Obstacle Distance
dist_obs = np.linalg.norm(traj - P_OBS, axis=1)
ax_obs.plot(time_steps_hist, dist_obs, 'r', linewidth=2)
ax_obs.axhline(R_OBS + R_SAFE, color='k', linestyle='--', label=f'Minimum Safe Distance ({R_OBS + R_SAFE}m)')
ax_obs.fill_between(time_steps_hist, 0, R_OBS + R_SAFE, color='red', alpha=0.1)
ax_obs.set_ylabel('Distance to Obstacle Center (m)')
ax_obs.set_title('Obstacle Avoidance Clearance')
ax_obs.legend()

# Docking Cone Angle Tracker (Relative to true target)
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
plt.show()