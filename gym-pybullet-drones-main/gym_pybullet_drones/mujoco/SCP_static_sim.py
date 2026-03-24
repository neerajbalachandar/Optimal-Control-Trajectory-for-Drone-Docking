import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time

# =====================================================================
# 1. SYSTEM CONSTANTS (Exactly from SCP_static_EKF.py)
# =====================================================================
dt_mpc = 0.1
N = 25  

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

A_d = np.eye(6)
A_d[0:3, 3:6] = dt_mpc * np.eye(3)
B_d = np.zeros((6, 3))
B_d[0:3, :] = 0.5 * dt_mpc**2 * np.eye(3)
B_d[3:6, :] = dt_mpc * np.eye(3)

# =====================================================================
# 2. MUJOCO & PD TRACKER SETUP
# =====================================================================
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

dt_sim = model.opt.timestep # 0.002s
STEPS_PER_MPC = int(dt_mpc / dt_sim) # 50 MuJoCo steps per 1 MPC step

MASS = 0.027
GRAVITY = 9.81

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp; self.kd = kd
    def compute(self, pos_err, vel_err):
        return self.kp * pos_err + self.kd * vel_err

pd_x = PDController(kp=3.0, kd=2.5)
pd_y = PDController(kp=3.0, kd=2.5)
pd_z = PDController(kp=8.0, kd=4.0) 

pd_roll  = PDController(kp=200.0, kd=50.0) 
pd_pitch = PDController(kp=200.0, kd=50.0)
pd_yaw   = PDController(kp=100.0, kd=25.0)

MAX_TILT = np.radians(25)
data.qpos[0:3] = x0[0:3]

# =====================================================================
# 3. VIEWER DRAWING FUNCTION
# =====================================================================
def draw_mpc_state(viewer, actual_path, mpc_path):
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

# =====================================================================
# 4. INITIAL WARM START (Like your theory code)
# =====================================================================
p_target = p_target_true.copy()
x_tar_full = np.hstack([p_target, np.zeros(3)])
x_rel_0 = x0 - x_tar_full

x_nom = np.zeros((N, 6))
for k in range(N):
    al = k / (N - 1)
    x_nom[k, 0:3] = (1 - al) * x_rel_0[0:3]
    x_nom[k, 1] += 0.5 * np.sin(np.pi * al) 

u_nom = np.zeros((N-1, 3))
phase = 0
MAX_ITERS = 2
TOL = 1e-3

log_actual_pos = []
sim_step_count = 0

print("\n[INFO] Starting EXACT True MPC Loop inside MuJoCo...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(1.0) # Let viewer open
    
    while viewer.is_running():
        
        # =========================================================
        # MPC STEP: RUNS EVERY 0.1 SECONDS (50 MuJoCo Steps)
        # =========================================================
        # 1. Get True State from MuJoCo Engine
        p_chaser = data.qpos[0:3].copy()
        v_chaser = data.qvel[0:3].copy()
        x_true = np.hstack([p_chaser, v_chaser])
        
        # 2. Add Sensor Noise to Target
        sensor_noise = np.random.normal(0, 0.02, 3) 
        sensor_noise = np.zeros(3) 
        
        p_target = p_target_true + sensor_noise
        x_tar_full = np.hstack([p_target, np.zeros(3)])
        x_rel_true = x_true - x_tar_full
        
        dist_to_goal = np.linalg.norm(x_true[0:3] - p_target_true)
        if dist_to_goal < 0.15 and np.linalg.norm(x_true[3:6]) < 0.2:
            print(f"Goal Reached!")
            break

        # 3. FSM Phase Logic
        dist_xy = np.linalg.norm(x_rel_true[0:2])
        if phase == 0 and dist_xy < 0.3:
            phase = 1
            print(f"[{sim_step_count * dt_mpc:.1f}s] FSM TRIGGER: Phase 1 (Docking Cone) Activated!")

        # 4. Warm-start shift
        if sim_step_count > 0:
            x_nom[:-1, :] = x_nom[1:, :]
            x_nom[-1, :] = x_nom[-2, :]
            u_nom[:-1, :] = u_nom[1:, :]
            u_nom[-1, :] = np.zeros(3)
            
        trust_radius = 2.0
        scp_converged = False
        
        # 5. SCP TRUST REGION LOOP (Exactly your math)
        t_start_solve = time.time()
        for it in range(MAX_ITERS):
            x_rel = cp.Variable((N, 6))
            u = cp.Variable((N-1, 3))
            
            slack_cone = cp.Variable(N-1, nonneg=True)
            slack_tar  = cp.Variable(N-1, nonneg=True)
            
            cost = 0
            con = [x_rel[0, :] == x_rel_true]
            
            # offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)
            # Slowly lower the terminal target so the solver doesn't panic
            offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.array([0.0, 0.0, 0.1])
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
            # prob.solve(solver=cp.ECOS, warm_start=True, ignore_dpp=True)
            # CLARABEL handles strict cones and high-weight slacks MUCH better than ECOS
            prob.solve(solver=cp.CLARABEL, warm_start=True, ignore_dpp=True)
            
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

        solve_time = time.time() - t_start_solve
        print(f"MPC Step {sim_step_count:03d} | Phase: {phase} | Cost: {prob.value:.1f} | Iters: {it+1} | Solve Time: {solve_time:.3f}s")

        # =========================================================
        # EXECUTE FIRST STEP (Inner Loop PD Tracking for 0.1s)
        # =========================================================
        # Extract target trajectory for the next 0.1s interval
        pos_start = p_chaser
        vel_start = v_chaser
        pos_end = x_nom[1, 0:3] + p_target # Convert relative back to global
        vel_end = x_nom[1, 3:6]

        # Update Viewer with the NEW "Thinking" Horizon Path
        mpc_global_path = x_nom[:, 0:3] + p_target
        with viewer.lock():
            draw_mpc_state(viewer, log_actual_pos, mpc_global_path)

        # Run 50 continuous physics steps to reach the next MPC state
        for pd_step in range(STEPS_PER_MPC):
            alpha = (pd_step + 1) / STEPS_PER_MPC
            
            # Smoothly interpolate targets over the 0.1s interval
            interp_pos = (1 - alpha) * pos_start + alpha * pos_end
            interp_vel = (1 - alpha) * vel_start + alpha * vel_end
            
            # Read Current State
            pos = data.qpos[0:3]
            quat = data.qpos[3:7] 
            lin_vel = data.qvel[0:3] 
            ang_vel = data.qvel[3:6] 
            
            w, xq, yq, zq = quat
            roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
            pitch = np.arcsin(2*(w*yq - zq*xq))
            yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))
            
            # Cascaded PD
            err_pos = interp_pos - pos
            err_vel = interp_vel - lin_vel
            
            acc_des_x = pd_x.compute(err_pos[0], err_vel[0])
            acc_des_y = pd_y.compute(err_pos[1], err_vel[1])
            acc_des_z = pd_z.compute(err_pos[2], err_vel[2])
            
            target_pitch = np.clip(acc_des_x / GRAVITY, -MAX_TILT, MAX_TILT)
            target_roll  = np.clip(-acc_des_y / GRAVITY, -MAX_TILT, MAX_TILT)
            target_yaw   = 0.0

            u_roll  = pd_roll.compute(target_roll - roll, 0.0 - ang_vel[0])
            u_pitch = pd_pitch.compute(target_pitch - pitch, 0.0 - ang_vel[1])
            u_yaw   = pd_yaw.compute(target_yaw - yaw, 0.0 - ang_vel[2])
            
            total_thrust = MASS * (GRAVITY + acc_des_z)
            data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
            data.ctrl[1] = -u_roll
            data.ctrl[2] = -u_pitch
            data.ctrl[3] = -u_yaw

            log_actual_pos.append(pos.copy())
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Since CVXPY takes time, we do NOT sleep here. 
            # We want the 50 frames to play out as fast as possible so the viewer doesn't feel completely frozen.

        sim_step_count += 1