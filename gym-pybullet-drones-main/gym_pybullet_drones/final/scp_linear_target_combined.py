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
PLAN_INTERVAL = 0.5     # Request a new plan every 0.5s
HORIZON = 15            # Look ahead steps
DT_PLAN = 0.1           # SCP Time step

# --- DOCKING CONFIGURATION ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    # Degrees
SAFETY_R     = 0.1   # Safety Radius (Hull size)
ALPHA_LIMIT  = 1.05  # Collision Trigger Threshold

# Wind
WIND_NOMINAL  = np.array([0.5, -0.3, -0.1]) 
WIND_GUST_AMP = 0.1                         

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. HELPER CLASSES (Target & EKF) - [PRESERVED]
# ======================================================================
class LinearTarget:
    def get_state(self, t):
        s = np.zeros(6)
        s[0] = 0.5 + 0.3 * t; s[1] = 0.0; s[2] = 1.0; s[3] = 0.3
        return s

class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6)*0.01; self.R = np.eye(3)*0.05
        self.P = np.eye(6)*0.1
    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x
    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        for i in range(steps): tmp = Fp @ tmp; fut[i,:] = tmp
        return fut

# ======================================================================
# 2. SAFETY & VISUALS - [UPDATED WITH CAPTURE VECTOR]
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    alpha_analytic = dist / (r1 + r2)
    return alpha_analytic

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
    
    # Basis
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    # 1. Draw Rim (Green)
    for phi in np.linspace(0, 2*np.pi, 12):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        p.addUserDebugLine(center, end, [0,1,0], 1, lifeTime=0.1, physicsClientId=client)
    
    # 2. Capture Vector / Center Axis (RED & THICK)
    p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 7, lifeTime=0.1, physicsClientId=client)

# ======================================================================
# 3. ASYNC PLANNER - [PRESERVED]
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True
        
        self.lock = threading.Lock()
        self.request_data = None 
        self.latest_plan = None  
        
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt
        self.prev_sol = None

    def trigger_update(self, chaser_state, target_preds):
        with self.lock:
            self.request_data = (chaser_state, target_preds)

    def get_plan(self):
        with self.lock:
            return self.latest_plan

    def run(self):
        print("[Planner Thread] Started.")
        while True:
            data = None
            with self.lock:
                if self.request_data:
                    data = self.request_data
                    self.request_data = None
            
            if data is None:
                time.sleep(0.01)
                continue

            chaser_st, preds = data
            try:
                traj = self._solve_scp(chaser_st, preds)
                with self.lock:
                    self.latest_plan = traj
            except Exception as e:
                print(f"Solver Error: {e}")

    def _solve_scp(self, chaser_st, preds):
        x_ref = np.zeros((6, self.N))
        if self.prev_sol is None:
            # Cold Start: Sine Curve Initialization
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*chaser_st[0:3] + al*preds[-1,0:3]
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) 
        else:
            self.prev_sol[:, :-1] = self.prev_sol[:, 1:]
            self.prev_sol[:, -1] = preds[-1]
            x_ref = self.prev_sol

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)
        
        for _ in range(2): 
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack_obs = cp.Variable(self.N, nonneg=True)
            slack_cone = cp.Variable(self.N, nonneg=True)
            
            cost = 0
            con = [x[:,0] == chaser_st]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # Tracking Cost 
                # Target offset: Stay 0.5m out along approach axis
                dock_pt = preds[k+1, 0:3] - n_app * 0.5
                cost += cp.sum_squares(x[0:3, k+1] - dock_pt) * 20
                cost += cp.sum_squares(u[:,k]) * 0.1
                
                # 1. Obstacle
                vec = x_ref[0:3,k+1] - self.p_obs
                n_obs = vec / (np.linalg.norm(vec) + 1e-4)
                con += [n_obs @ (x[0:3,k+1] - self.p_obs) >= self.r_obs - slack_obs[k+1]]
                
                # 2. Docking Cone
                p_rel = x[0:3,k+1] - preds[k+1, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k+1]]

                con += [cp.norm(x[:,k+1] - x_ref[:,k+1]) <= 0.5]

            cost += cp.sum(slack_obs)*10000 + cp.sum(slack_cone)*10000
            
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.CLARABEL)
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value
            
        self.prev_sol = x_ref
        return x_ref

# ======================================================================
# 4. MAIN RUN
# ======================================================================
def run():
    CHASER_START = np.array([-2.0, 0.0, 1.0])
    OBS_POS = np.array([-0.5, 0.0, 1.0]); OBS_RAD = 0.4
    
    target_gen = LinearTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    ekf.x = target_gen.get_state(0)
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, OBS_POS, OBS_RAD, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    # Environment
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, 
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, target_gen.get_state(0)[0:3]]),
        physics=Physics.PYB_DW, # Downwash Enabled
        neighbourhood_radius=0.3,
        pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=True, obstacles=False
    )
    PYB = env.getPyBulletClient()
    
    # Visuals: Obstacle
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RAD),
                      p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RAD, rgbaColor=[1,0,0,0.5]),
                      OBS_POS, physicsClientId=PYB)
    
    # Visuals: Hulls
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB) 
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB) 
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    # State Vars
    state = STATE_TRACKING
    current_traj = None
    last_plan_time = 0
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    print("[SIM] Running...")
    START = time.time()
    
    for i in range(int(20.0 * CTRL_FREQ)):
        sim_t = i / CTRL_FREQ
        
        # --- PHYSICS ---
        obs, _, _, _, _ = env.step(action)
        
        # Update Target
        true_state = target_gen.get_state(sim_t)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
        
        # Update Hulls
        p_chaser = obs[0][0:3]
        p_target = obs[1][0:3]
        update_hull(hull_c, p_chaser, PYB)
        update_hull(hull_t, p_target, PYB)
        
        # Wind
        gust = np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
        wind_total = WIND_NOMINAL + gust
        p.applyExternalForce(env.DRONE_IDS[0], -1, wind_total, p_chaser, p.WORLD_FRAME, PYB)
        
        # DRAW WIND (Yellow) & CAPTURE VECTOR (Red)
        p.addUserDebugLine(p_chaser, p_chaser + wind_total*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
        draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

        # --- ESTIMATION ---
        meas = true_state[0:3] + np.random.normal(0, 0.02, 3)
        ekf.step(meas)
        
        # --- LOGIC ---
        alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, OBS_POS, OBS_RAD)
        alpha_targ = solve_dcol_scaling(p_chaser, SAFETY_R, p_target, SAFETY_R)
        
        if state == STATE_TRACKING:
            if alpha_obs < ALPHA_LIMIT or alpha_targ < ALPHA_LIMIT:
                print(f"\n[ALERT] Safety Violation! Backing Off.")
                state = STATE_BACKING_OFF
                backoff_start = p_chaser.copy()
                vec = p_chaser - p_target
                vec = vec / (np.linalg.norm(vec)+1e-6)
                backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                backoff_t_start = time.time()
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
            
            else:
                new_plan = planner.get_plan()
                if new_plan is not None:
                    current_traj = new_plan
                    # Draw Path (Blue)
                    p.removeAllUserDebugItems(physicsClientId=PYB) 
                    for j in range(HORIZON-1):
                        p.addUserDebugLine(current_traj[0:3,j], current_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                if sim_t - last_plan_time > PLAN_INTERVAL:
                    preds = ekf.predict_future(HORIZON, DT_PLAN)
                    chaser_st = np.hstack([p_chaser, obs[0][10:13]])
                    planner.trigger_update(chaser_st, preds)
                    last_plan_time = sim_t
                
                if current_traj is not None:
                    pt = current_traj[0:3, 1] 
                    vt = current_traj[3:6, 1]
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
            
            if sim_t - last_plan_time > 0.1:
                preds = ekf.predict_future(HORIZON, DT_PLAN)
                chaser_st = np.hstack([p_chaser, obs[0][10:13]])
                planner.trigger_update(chaser_st, preds)
                last_plan_time = sim_t
                
            new_plan = planner.get_plan()
            if new_plan is not None:
                if np.linalg.norm(new_plan[0:3,0] - p_chaser) < 0.3:
                    print("[FSM] Replan Successful.")
                    current_traj = new_plan
                    p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                    state = STATE_TRACKING

        action[1] = np.zeros(4) 
        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()