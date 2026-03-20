import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================================
# 0. CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     # Fast planning (5Hz)
HORIZON = 20            # 2.0s Horizon
DT_PLAN = 0.1           
DURATION_SEC = 20.0     

# --- DOCKING & SAFETY ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  

# Wind
WIND_NOMINAL  = np.array([0.5, -0.3, -0.1]) 
WIND_GUST_AMP = 0.1                         

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. STATIC TARGET 
# ======================================================================
class StaticTarget:
    def get_state(self, t):
        s = np.zeros(6)
        s[0] = 0.5; s[1] = 0.0; s[2] = 1.0 
        return s

# ======================================================================
# 2. TARGET EKF (UPGRADED TO 9D: Pos, Vel, Acc per Section III-B)
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(9) # [px, py, pz, vx, vy, vz, ax, ay, az]
        
        # 9x9 State Transition Matrix
        self.F = np.eye(9)
        # Position += v*dt + 0.5*a*dt^2
        self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.F[0,6]=0.5*dt**2; self.F[1,7]=0.5*dt**2; self.F[2,8]=0.5*dt**2
        # Velocity += a*dt
        self.F[3,6]=dt; self.F[4,7]=dt; self.F[5,8]=dt
        
        # Measurement Matrix (We only measure position)
        self.H = np.zeros((3,9)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(3) * 0.05
        self.P = np.eye(9) * 0.1

    def step(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 9))
        tmp = self.x.copy()
        
        Fp = np.eye(9)
        Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        Fp[0,6]=0.5*dt_plan**2; Fp[1,7]=0.5*dt_plan**2; Fp[2,8]=0.5*dt_plan**2
        Fp[3,6]=dt_plan; Fp[4,7]=dt_plan; Fp[5,8]=dt_plan
        
        for i in range(steps): 
            tmp = Fp @ tmp
            fut[i,:] = tmp
        return fut

# ======================================================================
# 3. SAFETY & VISUALS
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    return dist / (r1 + r2)

def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)
    
def draw_dynamic_cone(center, axis, angle_deg, client):
    if not hasattr(draw_dynamic_cone, "ids"): draw_dynamic_cone.ids = []
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis 
    length = 1.0 
    theta = np.deg2rad(angle_deg)
    
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    line_idx = 0
    for phi in np.linspace(0, 2*np.pi, 30):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        if len(draw_dynamic_cone.ids) <= line_idx:
            uid = p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, physicsClientId=client)
            draw_dynamic_cone.ids.append(uid)
        else:
            p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
        line_idx += 1

    if len(draw_dynamic_cone.ids) <= line_idx:
        uid = p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, physicsClientId=client)
        draw_dynamic_cone.ids.append(uid)
    else:
        p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
    
# ======================================================================
# 4. ASYNC PLANNER (UPGRADED WITH Q/R MATRICES AND PHYSICAL BOUNDS)
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt

    def request(self, chaser, preds):
        with self.lock: self.req = (chaser, preds)
    def get(self):
        with self.lock: return self.res

    def run(self):
        print("[Planner] Thread Started.")
        while True:
            data = None
            with self.lock:
                if self.req: data = self.req; self.req = None
            if data:
                try:
                    traj = self._solve_scp(data[0], data[1])
                    with self.lock: self.res = traj
                except Exception as e: 
                    # print(f"SCP Error: {e}")
                    pass
            time.sleep(0.01)

    def _solve_scp(self, start, preds):
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*start[0:3] + al*preds[-1,0:3]
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) 
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = np.hstack([preds[-1, 0:3], preds[-1, 3:6]])

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        # --- PAPER ALIGNMENT: MATRICES AND BOUNDS ---
        Q_mat = np.diag([1, 1, 5, 1, 1, 1])  # Pos & Vel Weight Matrix (Eq 20)
        R_mat = np.diag([0.1, 0.1, 0.1])        # Control Effort Weight Matrix (Eq 20)
        V_MAX = 5.0                             # Velocity Bound (Eq 10)
        U_MAX = 15.0                            # Thrust/Control Bound (Eq 9)

        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack_obs = cp.Variable(self.N, nonneg=True)
            slack_cone = cp.Variable(self.N, nonneg=True)
            
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # 1. State Tracking Cost (Q Matrix)
                target_state = np.zeros(6)
                target_state[0:3] = preds[k+1, 0:3] # Position
                target_state[3:6] = preds[k+1, 3:6] # Velocity
                cost += cp.quad_form(x[:, k+1] - target_state, Q_mat)
                
                # 2. Control Effort Cost (R Matrix)
                cost += cp.quad_form(u[:,k], R_mat)

                # 3. Physical Bounds (Eq 9 & 10)
                con += [cp.norm(x[3:6, k+1]) <= V_MAX] 
                con += [cp.abs(u[0,k]) <= U_MAX, cp.abs(u[1,k]) <= U_MAX, cp.abs(u[2,k]) <= U_MAX]

            for k in range(1, self.N):
                # Obstacle
                vec = x_ref[0:3,k] - self.p_obs
                n = vec / (np.linalg.norm(vec)+1e-4)
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack_obs[k]]
                
                # Cone Constraint
                p_rel = x[0:3,k] - preds[k, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]
                
                # Trust Region
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += cp.sum(slack_obs)*1e6 + cp.sum(slack_cone)*1e5
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref

    
def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION_SEC = 20.0
    
    CHASER_START = np.array([-2.5, 0.0, 1.5])
    P_OBS        = np.array([-1.0, 0.0, 1.25]) 
    R_OBS        = 0.4
    
    target_gen = StaticTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    
    # Init perfect 9D State for EKF
    init_s = target_gen.get_state(0)
    init_9d = np.zeros(9); init_9d[0:6] = init_s 
    ekf.x = init_9d 
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, P_OBS, R_OBS, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, 
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, init_s[0:3]]),
        physics=Physics.PYB_DW, 
        pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=True, obstacles=False
    )
    PYB = env.getPyBulletClient()
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=R_OBS),
                      p.createVisualShape(p.GEOM_SPHERE, radius=R_OBS, rgbaColor=[1,0,0,0.2]),
                      P_OBS, physicsClientId=PYB)
    
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    state = STATE_TRACKING
    curr_traj = None
    last_plan = 0
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    docking_phase = 0 
    DOCK_OFFSET = np.array([-0.2, 0.0, 0.3]) 
    
    frozen = False
    freeze_pos_c = None
    freeze_pos_t = None
    
    history = {
        't': [], 
        'p_c': [], 'p_t': [], 
        'v_c': [], 'action': []
    }
    
    print("[SIM] Running Static Target SCP Docking...")
    START = time.time()
    
    p.addUserDebugLine(init_s[0:3], init_s[0:3] + [0,0,0.1], [0.5, 0, 0.5], 5, physicsClientId=PYB)
    
    try:
        for i in range(int(DURATION_SEC * CTRL_FREQ)):
            sim_t = i / CTRL_FREQ
            
            if not p.isConnected(physicsClientId=PYB):
                print("\n[USER] Window closed. Finishing...")
                break
                
            keys = p.getKeyboardEvents()
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                print("[SIM] Paused. Press SPACE to resume.")
                while True:
                    keys = p.getKeyboardEvents()
                    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                        print("[SIM] Resumed.")
                        break
                    env.render()
                    time.sleep(0.01)
                    
            if frozen:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], freeze_pos_c, [0,0,0,1], physicsClientId=PYB)
                p.resetBasePositionAndOrientation(env.DRONE_IDS[1], freeze_pos_t, [0,0,0,1], physicsClientId=PYB)
                env.render()
                sync(i, START, env.CTRL_TIMESTEP)
                continue
                
            obs, _, _, _, _ = env.step(action)
            true_state = target_gen.get_state(sim_t)
            
            p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
            
            p_chaser = obs[0][0:3]
            p_target = obs[1][0:3]
            update_hull(hull_c, p_chaser, PYB)
            update_hull(hull_t, p_target, PYB)
            
            wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
            p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
            p.addUserDebugLine(p_chaser, p_chaser + wind*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
            draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

            ekf.step(true_state[0:3])
            alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, P_OBS, R_OBS)
            
            if docking_phase == 0:
                xy_dist = np.linalg.norm(p_chaser[0:2] - p_target[0:2])
                z_dist = p_chaser[2] - p_target[2]
                if xy_dist < 0.3 and z_dist < 0.3:
                    docking_phase = 1
                    print(f"[PHASE] {sim_t:.2f}s | Switched to Terminal Phase (Diving in!)")

            if docking_phase == 1 and np.linalg.norm(p_chaser - p_target) < 0.2:
                frozen = True
                freeze_pos_c = p_chaser
                freeze_pos_t = p_target
                
                print(f"\n[COMPLETE] DOCKED SUCCESSFULLY!")
                print(f">>> Total Docking Time: {sim_t:.2f} seconds <<<")
                
                history['t'].append(sim_t)
                history['p_c'].append(obs[0][0:3].copy())
                history['p_t'].append(true_state[0:3].copy())
                history['v_c'].append(obs[0][10:13].copy())
                history['action'].append(action[0].copy())
                
                # Assume plot_performance is defined elsewhere in your file
                # plot_performance(history)
                continue

            if state == STATE_TRACKING:
                if alpha_obs < ALPHA_LIMIT:
                    state = STATE_BACKING_OFF
                    backoff_start = p_chaser.copy()
                    vec = p_chaser - P_OBS
                    vec = vec / (np.linalg.norm(vec)+1e-6)
                    backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                    backoff_t_start = time.time()
                    p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
                else:
                    res = planner.get()
                    if res is not None: 
                        curr_traj = res
                        for j in range(HORIZON-1):
                            p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                    if sim_t - last_plan > PLAN_INTERVAL:
                        preds = ekf.predict_future(HORIZON, DT_PLAN)
                        if docking_phase == 0:
                            preds[:, 0:3] += DOCK_OFFSET
                        
                        chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                        planner.request(chaser_st, preds)
                        last_plan = sim_t

                    if docking_phase == 1:
                        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_target, np.zeros(3))
                    else:
                        if curr_traj is not None:
                                idx = min(3, curr_traj.shape[1]-1)
                                pt = curr_traj[0:3, idx] 
                                vt = curr_traj[3:6, idx]
                                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
                        else:
                                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START, np.zeros(3))
            
            elif state == STATE_BACKING_OFF:
                elapsed = time.time() - backoff_t_start
                progress = min(elapsed / 2.0, 1.0)
                k = progress * progress * (3 - 2 * progress)
                setpoint = (1-k)*backoff_start + k*backoff_end
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], setpoint, np.zeros(3))
                if progress >= 1.0:
                    state = STATE_REPLANNING
                    
            elif state == STATE_REPLANNING:
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser, np.zeros(3))
                if sim_t - last_plan > 0.1:
                    preds = ekf.predict_future(HORIZON, DT_PLAN)
                    if docking_phase == 0: preds[:, 0:3] += DOCK_OFFSET
                    chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                    planner.request(chaser_st, preds)
                    last_plan = sim_t
                
                res = planner.get()
                if res is not None:
                    if np.linalg.norm(res[0:3,0] - p_chaser) < 0.3:
                        curr_traj = res
                        p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                        state = STATE_TRACKING
            
            action[1] = np.zeros(4)
            
            if i % 20 == 0:
                dist = np.linalg.norm(p_chaser - p_target)
                p_str = "OFFSET" if docking_phase == 0 else "TERMINAL"
                print(f"{sim_t:05.2f} | Phase: {p_str} | Dist: {dist:.2f}m")
                
            history['t'].append(sim_t)
            history['p_c'].append(obs[0][0:3].copy())
            history['p_t'].append(true_state[0:3].copy())
            history['v_c'].append(obs[0][10:13].copy())
            history['action'].append(action[0].copy())

            env.render()
            sync(i, START, env.CTRL_TIMESTEP)
            
    except Exception as e:
        print(f"\n[INFO] Simulation ended: {e}")
        
    finally:
        try: env.close()
        except: pass
        
        if not frozen:
            print("[PLOTS] Time expired. Generating Performance Plots...")
            # plot_performance(history)

if __name__ == "__main__":
    run()