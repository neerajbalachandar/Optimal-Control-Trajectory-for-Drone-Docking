import numpy as np
import cvxpy as cp
import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =====================================================================
# 1. EXACT THEORY CONSTANTS (From your SCP_static_EKF.py)
# =====================================================================
dt = 0.1
N = 25  # Planning horizon

U_MAX = 15.0
V_MAX = 5.0

P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)
N_APP = np.array([0, 0, -1])

r_dock = 0 

x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
p_target_true = np.array([1.0, 0.0, 1.0]) 

# DYNAMICS (Used ONLY for the planner's internal constraints now)
A_d = np.eye(6)
A_d[0:3, 3:6] = dt * np.eye(3)
B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
B_d[3:6, :] = dt * np.eye(3)

# =====================================================================
# 2. MUJOCO SETUP (The "Body")
# =====================================================================
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

dt_sim = model.opt.timestep # 0.002s
STEPS_PER_MPC = int(dt / dt_sim) # 50 steps per 0.1s MPC interval

MASS = 0.027
GRAVITY = 9.81

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp; self.kd = kd
    def compute(self, pos_err, vel_err):
        return self.kp * pos_err + self.kd * vel_err

pd_roll  = PDController(kp=200.0, kd=50.0) 
pd_pitch = PDController(kp=200.0, kd=50.0)
pd_yaw   = PDController(kp=100.0, kd=25.0)

MAX_TILT = np.radians(25)

# Teleport drone to initial state
data.qpos[0:3] = x0[0:3]

# =====================================================================
# 3. EXACT THEORY INITIALIZATION
# =====================================================================
p_target = p_target_true.copy()
x_tar_full = np.hstack([p_target, np.zeros(3)])
x_rel_0 = x0 - x_tar_full

x_nom = np.zeros((N, 6))
for k in range(N):
    al = k / (N - 1)
    x_nom[k, 0:3] = (1 - al) * x_rel_0[0:3]
    x_nom[k, 1] += 0.5 * np.sin(np.pi * al) 
    
    
log_actual_pos = []

    
def draw_mpc_state(viewer, actual_path):
    viewer.user_scn.ngeom = 0 
    
    # 1. Obstacle (Red)
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[R_OBS, 0, 0],
                            pos=P_OBS, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.4]))
        viewer.user_scn.ngeom += 1

    # 2. CURRENT MPC Horizon (Blue Dots) - Shows what the drone is "Thinking"
    # for p in mpc_path:
    #     if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
    #     mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
    #                         type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0],
    #                         pos=p, mat=np.eye(3).flatten(), rgba=np.array([0, 0, 1, 0.8]))
    #     viewer.user_scn.ngeom += 1

    # 3. Executed Path (Green Dots)
    for p in actual_path[::20]: 
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                            pos=p, mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.8]))
        viewer.user_scn.ngeom += 1

SIM_MAX_STEPS = 80
TOL = 1e-3
MAX_ITERS = 10 

u_nom = np.zeros((N-1, 3))
phase = 0 

# Logging for your exact plots
x_hist = [x0.copy()]
u_hist = []
phase_hist = []
time_steps_hist = [0.0]

print("\n[INFO] Running EXACT SCP_static_EKF inside MuJoCo...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Draw Obstacle in viewer
    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE, 
                        size=[R_OBS, 0, 0], pos=P_OBS, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.4]))
    viewer.user_scn.ngeom = 1
    
    time.sleep(1.0)
    
    # =================================================================
    # 4. EXACT THEORY LOOP (for sim_step in range(SIM_MAX_STEPS):)
    # =================================================================
    for sim_step in range(SIM_MAX_STEPS):
        
        # --- A. READ TRUE STATE FROM MUJOCO ---
        p_chaser = data.qpos[0:3].copy()
        v_chaser = data.qvel[0:3].copy()
        x_true = np.hstack([p_chaser, v_chaser])
        
        # --- B. EXACT THEORY NOISE & TARGET LOGIC ---
        sensor_noise = np.random.normal(0, 0.02, 3) 
        p_target = p_target_true + sensor_noise
        x_tar_full = np.hstack([p_target, np.zeros(3)])
        
        dist_to_goal = np.linalg.norm(x_true[0:3] - p_target_true)
        vel_mag = np.linalg.norm(x_true[3:6])
        if dist_to_goal < 0.15 and vel_mag < 0.2:
            print(f"Goal Reached at step {sim_step}!")
            break
            
        x_rel_true = x_true - x_tar_full

        dist_xy = np.linalg.norm(x_rel_true[0:2])
        if phase == 0 and dist_xy < 0.3:
            phase = 1
            print(f"[{sim_step*dt:.1f}s] FSM TRIGGER: Phase 1 Activated!")

        if sim_step > 0:
            x_nom[:-1, :] = x_nom[1:, :]
            x_nom[-1, :] = x_nom[-2, :]
            u_nom[:-1, :] = u_nom[1:, :]
            u_nom[-1, :] = np.zeros(3)
            
        trust_radius = 2.0
        scp_converged = False
        
        # --- C. EXACT SCP OPTIMIZATION (Unmodified) ---
        for it in range(MAX_ITERS):
            x_rel = cp.Variable((N, 6))
            u = cp.Variable((N-1, 3))
            
            slack_cone = cp.Variable(N-1, nonneg=True)
            slack_tar  = cp.Variable(N-1, nonneg=True)
            
            cost = 0
            con = [x_rel[0, :] == x_rel_true]
            
            offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
            con += [x_rel[-1, 0:3] == offset] 
            con += [x_rel[-1, 3:6] == np.zeros(3)]
            
            for k in range(N-1):
                con += [x_rel[k+1, :] == A_d @ x_rel[k, :] + B_d @ u[k, :]]
                
                cost += cp.sum_squares(x_rel[k, 0:3] - offset) * 2.0 
                cost += cp.sum_squares(x_rel[k, 3:6]) * 1.0 
                cost += cp.sum_squares(u[k, :]) * 0.5
                cost += cp.sum_squares(u[k, :] - u_nom[k, :]) * 2.0
                
                con += [cp.norm(x_rel[k, :] - x_nom[k, :], np.inf) <= trust_radius]
                
                p_rel_nom = x_nom[k, 0:3]
                p_obs_rel = P_OBS - p_target
                v_obs = p_rel_nom - p_obs_rel
                d_obs = np.linalg.norm(v_obs) + 1e-8
                n_obs = v_obs / d_obs
                con += [n_obs @ (x_rel[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

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
            prob.solve(solver=cp.ECOS, warm_start=True)
            
            if prob.status != "optimal":
                trust_radius *= 0.5
                continue
                
            delta = np.linalg.norm(x_rel.value - x_nom, np.inf)
            x_nom = x_rel.value.copy()
            u_nom = u.value.copy() 
            
            if delta < TOL:
                scp_converged = True
                break
            trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

        print(f"Step {sim_step:02d} | Phase: {phase} | Cost: {prob.value:.1f} | Iters: {it+1}")

        # --- D. APPLY THEORY 'u' LIVE INTO MUJOCO ---
        # The theory outputs 'u' as a pure [a_x, a_y, a_z] acceleration vector.
        u_opt = u_nom[0, :]
        
        with viewer.lock():
            draw_mpc_state(viewer, log_actual_pos)
        
        for _ in range(STEPS_PER_MPC):
            
            quat = data.qpos[3:7] 
            ang_vel = data.qvel[3:6] 
            pos = data.qpos[0:3]
            
            
            w, xq, yq, zq = quat
            roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
            pitch = np.arcsin(2*(w*yq - zq*xq))
            yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))
            
            # Direct mapping from Theory Acceleration (u) to Drone Tilt
            target_pitch = np.clip(u_opt[0] / GRAVITY, -MAX_TILT, MAX_TILT)
            target_roll  = np.clip(-u_opt[1] / GRAVITY, -MAX_TILT, MAX_TILT)
            target_yaw   = 0.0

            # Inner-loop Attitude PD to force MuJoCo to tilt
            u_roll  = pd_roll.compute(target_roll - roll, 0.0 - ang_vel[0])
            u_pitch = pd_pitch.compute(target_pitch - pitch, 0.0 - ang_vel[1])
            u_yaw   = pd_yaw.compute(target_yaw - yaw, 0.0 - ang_vel[2])
            
            total_thrust = MASS * (GRAVITY + u_opt[2])
            data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
            data.ctrl[1] = -u_roll
            data.ctrl[2] = -u_pitch
            data.ctrl[3] = -u_yaw
            
            log_actual_pos.append(pos.copy())
            

            mujoco.mj_step(model, data)
            viewer.sync()

        # --- E. LOGGING ---
        x_hist.append(x_true.copy())
        u_hist.append(u_opt.copy())
        phase_hist.append(phase)
        time_steps_hist.append((sim_step + 1) * dt)

# =====================================================================
# 5. EXACT THEORY PLOTTING (Unmodified from your code)
# =====================================================================
x_hist = np.array(x_hist)
phase_hist = np.array(phase_hist)
time_steps_hist = np.array(time_steps_hist)

plt.style.use('seaborn-v0_8-darkgrid')

fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(121, projection='3d')
traj = x_hist[:, 0:3]
ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b.-', linewidth=3, label='Executed Traj')
ax1.plot(x0[0], x0[1], x0[2], 'go', markersize=8, label='Start')
ax1.plot(p_target_true[0], p_target_true[1], p_target_true[2], 'r*', markersize=12, label='True Target')

u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3)
ax1.set_title('Online Closed-Loop Trajectory')
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

plt.tight_layout()
plt.show()