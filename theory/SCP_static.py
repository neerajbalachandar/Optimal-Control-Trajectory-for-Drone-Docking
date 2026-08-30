import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ================= SYSTEM =================
dt = 0.1
N = 25
time_steps = np.arange(N) * dt
ctrl_steps = np.arange(N-1) * dt

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

x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
p_target = np.array([0.5, 0.0, 1.0])

# ================= DYNAMICS =================
A_d = np.eye(6)
A_d[0:3, 3:6] = dt * np.eye(3)

B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
B_d[3:6, :] = dt * np.eye(3)

# ================= DCOL (Tracy et al.) =================
def alpha(pc):
    return np.linalg.norm(pc - p_target) / (r_c + r_t)

def grad_alpha(pc):
    v = pc - p_target
    d = np.linalg.norm(v) + 1e-8
    return v / (d * (r_c + r_t))

# ================= INITIAL GUESS =================
x_nom = np.zeros((N, 6))
for k in range(N):
    al = k / (N - 1)
    x_nom[k, 0:3] = (1 - al) * x0[0:3] + al * p_target
    # Bow laterally AND slightly vertically to avoid diving under the rock
    x_nom[k, 1] += 0.8 * np.sin(np.pi * al) 
    x_nom[k, 2] += 0.5 * np.sin(np.pi * al)
    
    
# ================= TRACKING ARRAYS =================
cost_history = []
delta_history = []
trust_history = []
u_nom = np.zeros((N-1, 3))

# ================= SCP LOOP =================
trust_radius = 2.0
TOL = 1e-3
MAX_ITERS = 15

print("Starting SCP...")
for it in range(MAX_ITERS):
    x = cp.Variable((N, 6))
    u = cp.Variable((N-1, 3))
    
    # THE FIX: Add Slack variables for the Cone and Target DCOL (Eq. 20)
    slack_cone = cp.Variable(N-1, nonneg=True)
    slack_tar  = cp.Variable(N-1, nonneg=True)
    
    cost = 0
    con = [x[0, :] == x0]
    
    # Terminal Boundary Condition
    con += [x[-1, 0:3] == p_target] # (Or p_goal if you kept the hover fix)
    con += [x[-1, 3:6] == np.zeros(3)]
    
    for k in range(N-1):
        con += [x[k+1, :] == A_d @ x[k, :] + B_d @ u[k, :]]
        
        # Objective: Tracking Error + Control Effort
        cost += cp.sum_squares(x[k, 0:3] - p_target) * 10.0
        cost += cp.sum_squares(u[k, :]) * 1.0
        cost += cp.sum_squares(x[k, 3:6]) * 1.0 
        
        # Trust Region
        con += [cp.norm(x[k, :] - x_nom[k, :], np.inf) <= trust_radius]
        
        p_nom = x_nom[k, 0:3]
        
        # 3. Obstacle Avoidance (Linearized)
        v_obs = p_nom - P_OBS
        d_obs = np.linalg.norm(v_obs) + 1e-8
        n_obs = v_obs / d_obs
        con += [n_obs @ (x[k, 0:3] - P_OBS) >= R_OBS + R_SAFE]

        p_rel_nom = x_nom[k, 0:3]
        
        # A. Obstacle Avoidance (Obstacle is moving relative to us! ALWAYS ACTIVE)
        p_obs_rel = P_OBS
        v_obs = p_rel_nom - p_obs_rel
        d_obs = np.linalg.norm(v_obs) + 1e-8
        n_obs = v_obs / d_obs
        con += [n_obs @ (x[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

        # B & C. Target DCOL and Docking Cone (SPATIAL THRESHOLDING)
        # In relative space, distance to target is simply the norm of p_rel_nom!
        dist_xy = np.linalg.norm(p_rel_nom[0:2])
        
        if dist_xy < 1.5:
            # --- PHASE 1 (TERMINAL DIVE): Turn ON Safety Constraints ---
            
            # B. DCOL Collision Avoidance (Target is fixed at origin [0,0,0])
            dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
            n_tar = (p_rel_nom / dist_tar_nom) * (r_c + r_t)
            con += [n_tar @ x[k, 0:3] >= (r_c + r_t) * alpha_min - slack_tar[k]]
            
            # C. Docking Cone (Target is fixed at origin [0,0,0])
            con += [cp.norm(x[k, 0:3]) * np.cos(THETA) <= -N_APP @ x[k, 0:3] + slack_cone[k]]
        else:
            # --- PHASE 0 (APPROACH): Turn OFF Safety Constraints ---
            con += [slack_tar[k] == 0]
            con += [slack_cone[k] == 0]
            
        # 5. Physical Limits
        con += [cp.norm(u[k, :], np.inf) <= U_MAX]
        con += [cp.norm(x[k+1, 3:6], 2) <= V_MAX]

    con += [cp.norm(x[-1, :] - x_nom[-1, :], np.inf) <= trust_radius]
    
    # THE FIX: Penalize the slacks heavily so the drone obeys the cone/DCOL but doesn't crash the math
    cost += cp.sum(slack_cone) * 200.0
    cost += cp.sum(slack_tar) * 200.0

    prob = cp.Problem(cp.Minimize(cost), con)
    prob.solve(solver=cp.ECOS)
    
    if prob.status != "optimal":
        print(f"Iter {it}: Infeasible! Shrinking trust region.")
        trust_radius *= 0.5
        continue
        
    delta = np.linalg.norm(x.value - x_nom, np.inf)
    
    # Store history for plotting
    cost_history.append(prob.value)
    delta_history.append(delta)
    trust_history.append(trust_radius)
    
    print(f"Iter {it} | Cost: {prob.value:.1f} | delta: {delta:.4f} | trust: {trust_radius:.4f}")
    
    x_nom = x.value.copy()
    u_nom = u.value.copy() # Save the optimized control!
    
    if delta < TOL:
        print(">>> SCP CONVERGED SUCCESSFULLY <<<")
        break
        
    trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)


# ====================================================================
# ========================= PLOTTING DASHBOARDS ======================
# ====================================================================

plt.style.use('seaborn-v0_8-darkgrid')

# ----------------- FIGURE 1: 3D Trajectory -----------------
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
traj = x_nom[:, 0:3]
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b.-', linewidth=3, label='Optimized Chaser Traj')
ax1.plot(x0[0], x0[1], x0[2], 'go', markersize=8, label='Start')
ax1.plot(p_target[0], p_target[1], p_target[2], 'r*', markersize=12, label='Target')

# Obstacle Sphere
u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3)

# Target Safety Bubble
x_tar = p_target[0] + (r_c+r_t)*np.cos(u_sph)*np.sin(v_sph)
y_tar = p_target[1] + (r_c+r_t)*np.sin(u_sph)*np.sin(v_sph)
z_tar = p_target[2] + (r_c+r_t)*np.cos(v_sph)
ax1.plot_surface(x_tar, y_tar, z_tar, color='g', alpha=0.2)
ax1.set_title('DCOL-Enforced Drone Trajectory')
ax1.legend()


# ----------------- FIGURE 2: Kinematics & Control -----------------
fig2, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Control Plot
ax_u.plot(ctrl_steps, u_nom[:, 0], 'r', label='$u_x$')
ax_u.plot(ctrl_steps, u_nom[:, 1], 'g', label='$u_y$')
ax_u.plot(ctrl_steps, u_nom[:, 2], 'b', label='$u_z$')
ax_u.axhline(U_MAX, color='k', linestyle='--', label='$+U_{max}$ (Limit)')
ax_u.axhline(-U_MAX, color='k', linestyle='--')
ax_u.set_ylabel('Control Acceleration ($m/s^2$)')
ax_u.set_title('Control Inputs vs. Time (Equation 9)')
ax_u.legend(loc='upper right')

# Velocity Plot
v_norms = np.linalg.norm(x_nom[:, 3:6], axis=1)
ax_v.plot(time_steps, v_norms, 'purple', linewidth=2, label='$||v||_2$')
ax_v.axhline(V_MAX, color='k', linestyle='--', label='$V_{max}$ (Limit)')
ax_v.set_xlabel('Time (s)')
ax_v.set_ylabel('Velocity Magnitude ($m/s$)')
ax_v.set_title('Velocity Profile vs. Time (Equation 10)')
ax_v.legend(loc='upper right')


# ----------------- FIGURE 3: Safety Constraints -----------------
fig3, (ax_obs, ax_tar, ax_cone) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Obstacle Distance
dist_obs = np.linalg.norm(traj - P_OBS, axis=1)
ax_obs.plot(time_steps, dist_obs, 'r', linewidth=2)
ax_obs.axhline(R_OBS + R_SAFE, color='k', linestyle='--', label=f'Minimum Safe Distance ({R_OBS + R_SAFE}m)')
ax_obs.fill_between(time_steps, 0, R_OBS + R_SAFE, color='red', alpha=0.1)
ax_obs.set_ylabel('Distance to Obstacle Center (m)')
ax_obs.set_title('Obstacle Avoidance Clearance')
ax_obs.legend()

# Target Alpha (DCOL)
alpha_history = [alpha(p) for p in traj]
ax_tar.plot(time_steps, alpha_history, 'g', linewidth=2)
ax_tar.axhline(alpha_min, color='k', linestyle='--', label=f'$\\alpha_{{min}}$ Trigger ({alpha_min})')
ax_tar.axhline(1.0, color='red', linestyle='-', label='Collision ($\\alpha = 1.0$)')
ax_tar.fill_between(time_steps, 0, 1.0, color='red', alpha=0.1)
ax_tar.set_xlabel('Time (s)')
ax_tar.set_ylabel('DCOL $\\alpha$ Scale')
ax_tar.set_title('Target DCOL Safety Factor (Phase 0)')
ax_tar.legend()

# 3. Docking Cone Angle Tracker
angles = []
for p_rel in x_nom[:, 0:3]:
    dist = np.linalg.norm(p_rel)
    if dist < 1e-5:
        angles.append(0.0)
    else:
        # Cosine rule for angle between approach axis and relative position
        cos_phi = np.clip(np.dot(-N_APP, p_rel) / dist, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_phi)))
        
ax_cone.plot(time_steps, angles, 'c', linewidth=2)
ax_cone.axhline(np.degrees(THETA), color='k', linestyle='--', label=f'Cone Limit ({np.degrees(THETA)}°)')
ax_cone.fill_between(time_steps, 0, np.degrees(THETA), color='cyan', alpha=0.1)
ax_cone.set_ylabel('Approach Angle (deg)')
ax_cone.set_xlabel('Time (s)')
ax_cone.set_title('Docking Cone Alignment vs. Time')
ax_cone.legend()



# ----------------- FIGURE 4: Solver Convergence -----------------
fig4, (ax_c,ax_r, ax_d, ax_t) = plt.subplots(1, 4, figsize=(15, 4))
iters = range(1, len(cost_history)+1)

ax_c.plot(iters, cost_history, 'mo-', linewidth=2)
ax_c.set_title('Objective Cost')
ax_c.set_xlabel('Iteration')
ax_c.set_ylabel('Cost')

dist_to_target = np.linalg.norm(x_nom[:, 0:3] - p_target, axis=1)

ax_r.plot(time_steps, dist_to_target, 'mo-', linewidth=2)
ax_r.set_title('Distance to Target vs Time')
ax_r.set_xlabel('Time (s)')
ax_r.set_ylabel('Distance (m)')

ax_d.semilogy(iters, delta_history, 'co-', linewidth=2)
ax_d.axhline(TOL, color='k', linestyle='--', label='Tolerance')
ax_d.set_title('Max Trajectory Change ($\\delta$)')
ax_d.set_xlabel('Iteration')
ax_d.set_ylabel('$\\delta$ (Log Scale)')
ax_d.legend()

ax_t.plot(iters, trust_history, 'yo-', linewidth=2)
ax_t.set_title('Trust Region Radius')
ax_t.set_xlabel('Iteration')
ax_t.set_ylabel('Radius (m)')

plt.tight_layout()
plt.show()