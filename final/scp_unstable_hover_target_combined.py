import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ======================================================================
# 0. CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     
HORIZON = 20            
DT_PLAN = 0.1           
DURATION_SEC = 20.0     # Longer duration to watch the slow fall
DROP_TIME = 1.0         # Time when battery starts failing

# --- SAFETY ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  

# Wind
WIND_NOMINAL  = np.array([0.2, -0.1, 0.0]) 
WIND_GUST_AMP = 0.05                         

# FSM
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. SLOW BATTERY FAILURE TARGET (Physics)
# ======================================================================
class BatteryFailureTarget:
    def get_state(self, t):
        s = np.zeros(9)
        
        # Initial State
        x_0 = 0.5
        y_0 = 0.5
        z_0 = 3.0 # Start higher
        
        if t < DROP_TIME:
            # Phase 1: Stable Hover
            s[0] = x_0; s[1] = y_0; s[2] = z_0
            s[3] = 0.0; s[4] = 0.0; s[5] = 0.0
            s[6] = 0.0; s[7] = 0.0; s[8] = 0.0
        else:
            # Phase 2: Slow Fall (Brownout)
            dt = t - DROP_TIME
            
            # Use small gravity to simulate partial power loss
            # a_z = -0.3 m/s^2 (Very slow acceleration down)
            g_eff = 0.5
            
            # Pos (z = z0 - 0.5*a*t^2)
            s[0] = x_0
            s[1] = y_0
            s[2] = z_0 - 0.5 * g_eff * dt**2
            
            # Vel (v = -a*t)
            s[3] = 0.0
            s[4] = 0.0
            s[5] = -g_eff * dt
            
            # Acc
            s[6] = 0.0
            s[7] = 0.0
            s[8] = -g_eff 
            
        return s

# ======================================================================
# 2. INTELLIGENT EKF (9-State)
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(9)
        
        # Constant Acceleration Model
        self.F = np.eye(9)
        self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.F[0,6]=0.5*dt**2; self.F[1,7]=0.5*dt**2; self.F[2,8]=0.5*dt**2
        self.F[3,6]=dt; self.F[4,7]=dt; self.F[5,8]=dt
        
        self.H = np.zeros((3,9)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        
        # Tuning: High Uncertainty in Acceleration change
        self.Q = np.eye(9) * 0.001
        self.Q[6:9, 6:9] *= 100.0 
        
        self.R = np.eye(3) * 0.01 
        self.P = np.eye(9) * 0.1

    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.x
    
    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6))
        tmp = self.x.copy()
        
        Fp = np.eye(9)
        Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        Fp[0,6]=0.5*dt_plan**2; Fp[1,7]=0.5*dt_plan**2; Fp[2,8]=0.5*dt_plan**2
        Fp[3,6]=dt_plan; Fp[4,7]=dt_plan; Fp[5,8]=dt_plan
        
        for i in range(steps):
            tmp = Fp @ tmp
            fut[i, 0:3] = tmp[0:3]
            fut[i, 3:6] = tmp[3:6]
        return fut

# ======================================================================
# 3. SAFETY HELPERS
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
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis
    length = 1.5
    theta = np.deg2rad(angle_deg)
    
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    # Rim
    for phi in np.linspace(0, 2*np.pi, 12):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        p.addUserDebugLine(center, end, [0,1,0], 1, lifeTime=0.1, physicsClientId=client)
    # Capture Vector (Red)
    p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 4, lifeTime=0.1, physicsClientId=client)

# ======================================================================
# 4. ASYNC PLANNER
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
                except: pass
            time.sleep(0.01)

    def _solve_scp(self, start, preds):
        # Warm Start
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3, k] = (1-al)*start[0:3] + al*preds[-1,0:3]
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = preds[-1]

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack_cone = cp.Variable(self.N, nonneg=True)
            
            dock_offset = -n_app * 0.1
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # --- FIXED: Variable Name Typo Corrected ---
                pos_err = x[0:3, k+1] - (preds[k+1,0:3] + dock_offset)
                cost += 50 * cp.sum_squares(pos_err[0:2]) # XY
                cost += 200 * cp.square(pos_err[2])       # Z (Priority!)
                cost += 0.001 * cp.sum_squares(u[:,k])

            # Constraints (Just Cone for open space fall)
            for k in range(1, self.N):
                p_rel = x[0:3,k] - preds[k, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += cp.sum(slack_cone)*1e5
            
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref

# ======================================================================
# 5. MAIN
# ======================================================================
def run():
    # Setup: Chaser starts higher
    CHASER_START = np.array([-1.5, 0.5, 3.5]) 
    OBS_POS = np.array([100, 100, 100]); OBS_RAD = 0.1 
    
    target_gen = BatteryFailureTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    init_s = target_gen.get_state(0)
    ekf.x[0:3] = init_s[0:3]
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, OBS_POS, OBS_RAD, CONE_ANGLE, DOCKING_AXIS)
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
    
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    state = STATE_TRACKING
    curr_traj = None
    last_plan = 0
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    print("[SIM] Running Slow Battery Failure Scenario...")
    START = time.time()
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        sim_t = i / CTRL_FREQ
        
        obs, _, _, _, _ = env.step(action)
        true_state = target_gen.get_state(sim_t)
        
        # Stop at floor
        if true_state[2] < 0.05: 
            true_state[2] = 0.05
            true_state[3:6] = 0 
            
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
        
        # Visuals
        p_chaser = obs[0][0:3]
        p_target = obs[1][0:3]
        update_hull(hull_c, p_chaser, PYB)
        update_hull(hull_t, p_target, PYB)
        
        wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
        p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
        p.addUserDebugLine(p_chaser, p_chaser + wind*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
        draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

        # EKF
        ekf.step(true_state[0:3] + np.random.normal(0, 0.01, 3))
        
        # FSM
        alpha_targ = solve_dcol_scaling(p_chaser, SAFETY_R, p_target, SAFETY_R)
        
        if state == STATE_TRACKING:
            dist = np.linalg.norm(p_chaser - p_target)
            p_rel = p_chaser - p_target
            misalignment = np.linalg.norm(np.cross(p_rel, DOCKING_AXIS))
            
            if alpha_targ < ALPHA_LIMIT and misalignment > 0.2:
                print(f"[ALERT] Unsafe Approach! Backing Off.")
                state = STATE_BACKING_OFF
                backoff_start = p_chaser.copy()
                vec = p_chaser - p_target
                vec = vec / (np.linalg.norm(vec)+1e-6)
                backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                backoff_t_start = time.time()
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
            
            else:
                # Plan
                res = planner.get()
                if res is not None: 
                    curr_traj = res
                    p.removeAllUserDebugItems(physicsClientId=PYB)
                    for j in range(HORIZON-1):
                        p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                if sim_t - last_plan > PLAN_INTERVAL:
                    preds = ekf.predict_future(HORIZON, DT_PLAN)
                    chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                    planner.request(chaser_st, preds)
                    last_plan = sim_t
                    
                    for j in range(HORIZON-1):
                        p.addUserDebugLine(preds[j,0:3], preds[j+1,0:3], [0,1,0], 2, physicsClientId=PYB)

                # Control
                if curr_traj is not None:
                    idx = 2 if 2 < curr_traj.shape[1] else 0
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
                print("[FSM] Safe. Replanning...")
                state = STATE_REPLANNING
                
        elif state == STATE_REPLANNING:
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser, np.zeros(3))
            if sim_t - last_plan > 0.1:
                preds = ekf.predict_future(HORIZON, DT_PLAN)
                chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                planner.request(chaser_st, preds)
                last_plan = sim_t
            
            res = planner.get()
            if res is not None:
                if np.linalg.norm(res[0:3,0] - p_chaser) < 0.3:
                    print("[FSM] Replan Success.")
                    curr_traj = res
                    p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                    state = STATE_TRACKING
        
        action[1] = np.zeros(4)
        
        if i % 20 == 0:
            c_pos = obs[0][0:3]
            t_pos = obs[1][0:3]
            status = "HOVER" if sim_t < DROP_TIME else "FALLING!"
            print(f"{sim_t:05.2f}  | {c_pos[2]:+.2f}      | {t_pos[2]:+.2f}      | {status}")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()