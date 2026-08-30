import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time
from scipy.interpolate import interp1d

# =====================================================================
# 1. EXACT SCP PLANNER (Static Target)
# =====================================================================
P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1
x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
p_target_true = np.array([0.5, 0.0, 1.0]) 

def plan_offline_trajectory():
    print("[INFO] Running SCP Planner...")
    dt = 0.1
    N = 25
    U_MAX = 15.0
    V_MAX = 5.0

    A_d = np.eye(6)
    A_d[0:3, 3:6] = dt * np.eye(3)
    B_d = np.zeros((6, 3))
    B_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
    B_d[3:6, :] = dt * np.eye(3)

    x_nom = np.zeros((N, 6))
    for k in range(N):
        al = k / (N - 1)
        x_nom[k, 0:3] = x0[0:3] + al * (p_target_true - x0[0:3])
        x_nom[k, 1] += 0.5 * np.sin(np.pi * al) 

    trust_radius = 2.0

    for it in range(5): 
        x = cp.Variable((N, 6))
        u = cp.Variable((N-1, 3))
        cost = 0
        con = [x[0, :] == x0]
        con += [x[-1, 0:3] == p_target_true] 
        con += [x[-1, 3:6] == np.zeros(3)]

        for k in range(N-1):
            con += [x[k+1, :] == A_d @ x[k, :] + B_d @ u[k, :]]
            cost += cp.sum_squares(u[k, :]) * 0.5
            con += [cp.norm(x[k, :] - x_nom[k, :], np.inf) <= trust_radius]
            con += [cp.norm(u[k, :], np.inf) <= U_MAX]
            con += [cp.norm(x[k+1, 3:6], 2) <= V_MAX]

            p_rel_nom = x_nom[k, 0:3]
            v_obs = p_rel_nom - P_OBS
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (x[k, 0:3] - P_OBS) >= R_OBS + R_SAFE]

        prob = cp.Problem(cp.Minimize(cost), con)
        prob.solve(solver=cp.ECOS)
        
        delta = np.linalg.norm(x.value - x_nom, np.inf)
        x_nom = x.value.copy()
        
        if delta < 1e-3: break

    print("[INFO] SCP Converged!")
    return x_nom

x_scp_discrete = plan_offline_trajectory()

# =====================================================================
# 2. INTERPOLATE SCP TO MUJOCO FREQUENCY
# =====================================================================
scp_times = np.linspace(0, 2.5, 25) 
dt_sim = 0.002
sim_times = np.arange(0, 5.0, dt_sim)

interp_pos = interp1d(scp_times, x_scp_discrete[:, 0:3], axis=0, kind='cubic', bounds_error=False, fill_value=(x_scp_discrete[0, 0:3], x_scp_discrete[-1, 0:3]))
interp_vel = interp1d(scp_times, x_scp_discrete[:, 3:6], axis=0, kind='cubic', bounds_error=False, fill_value=(x_scp_discrete[0, 3:6], np.zeros(3)))

trajectory_pos = interp_pos(sim_times)
trajectory_vel = interp_vel(sim_times)

# =====================================================================
# 3. MUJOCO TRACKING CONTROLLER WITH IN-SIM RENDERING
# =====================================================================
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

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
data.qpos[0:3] = x_scp_discrete[0, 0:3]

log_actual_pos = []

# --- Custom Draw Function ---
def draw_custom_visuals(viewer, actual_path):
    viewer.user_scn.ngeom = 0 # Reset frame geometry
    
    # 1. Draw Obstacle (Red Sphere)
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[R_OBS, 0, 0],
                            pos=P_OBS, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.4]))
        viewer.user_scn.ngeom += 1

    # 2. Draw SCP Planned Path (Blue Dots)
    for p in x_scp_discrete[:, 0:3]:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0],
                            pos=p, mat=np.eye(3).flatten(), rgba=np.array([0, 0, 1, 0.8]))
        viewer.user_scn.ngeom += 1

    # 3. Draw Actual Drone Path Trail (Green Dots)
    # Subsample the trajectory so we don't blow up the geometry count
    for p in actual_path[::20]: 
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                            pos=p, mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.8]))
        viewer.user_scn.ngeom += 1

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_idx = 0
    total_steps = len(sim_times)
    
    time.sleep(1.0)
    
    while viewer.is_running() and step_idx < total_steps:
        step_start = time.time()
        
        pos = data.qpos[0:3].copy()
        quat = data.qpos[3:7] 
        lin_vel = data.qvel[0:3] 
        ang_vel = data.qvel[3:6] 
        
        w, xq, yq, zq = quat
        roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
        pitch = np.arcsin(2*(w*yq - zq*xq))
        yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))
        
        target_pos = trajectory_pos[step_idx]
        target_vel = trajectory_vel[step_idx]
            
        err_pos = target_pos - pos
        err_vel = target_vel - lin_vel
        
        acc_des_x = pd_x.compute(err_pos[0], err_vel[0])
        acc_des_y = pd_y.compute(err_pos[1], err_vel[1])
        acc_des_z = pd_z.compute(err_pos[2], err_vel[2])
        
        target_pitch = acc_des_x / GRAVITY
        target_roll  = -acc_des_y / GRAVITY
        target_yaw   = 0.0
        
        target_pitch = np.clip(target_pitch, -MAX_TILT, MAX_TILT)
        target_roll  = np.clip(target_roll, -MAX_TILT, MAX_TILT)

        err_roll  = target_roll - roll
        err_pitch = target_pitch - pitch
        err_yaw   = target_yaw - yaw
        
        u_roll  = pd_roll.compute(err_roll, 0.0 - ang_vel[0])
        u_pitch = pd_pitch.compute(err_pitch, 0.0 - ang_vel[1])
        u_yaw   = pd_yaw.compute(err_yaw, 0.0 - ang_vel[2])
        
        total_thrust = MASS * (GRAVITY + acc_des_z)
        data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
        data.ctrl[1] = -u_roll
        data.ctrl[2] = -u_pitch
        data.ctrl[3] = -u_yaw

        log_actual_pos.append(pos)
        
        # Inject our custom shapes into the MuJoCo viewer!
        with viewer.lock():
            draw_custom_visuals(viewer, log_actual_pos)

        mujoco.mj_step(model, data)
        viewer.sync()
        step_idx += 1
        
        time_until_next_step = dt_sim - (time.time() - step_start)
        if time_until_next_step > 0: time.sleep(time_until_next_step)