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
# CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     # 5Hz Planning (Fast Reflexes needed for wobble)
HORIZON = 20            # 2.0s Horizon
DT_PLAN = 0.1           
DURATION_SEC = 20.0     

# ======================================================================
# 1. GROUND TRUTH (UNSTABLE HOVER TARGET)
# ======================================================================
class UnstableHoverTarget:
    def get_state(self, t):
        s = np.zeros(6)
        
        # --- PHYSICS ---
        z_start = 2.5
        descent_rate = 0.2  # Slow descent
        wobble_freq = 2.0   # Rad/s
        wobble_amp = 0.3    # Meters
        
        # Position (Spiral/Wobble down)
        s[0] = wobble_amp * np.sin(wobble_freq * t) # X wobble
        s[1] = wobble_amp * np.cos(wobble_freq * t) # Y wobble
        s[2] = z_start - descent_rate * t           # Linear Descent
        
        # Velocity (Derivative)
        s[3] = wobble_amp * wobble_freq * np.cos(wobble_freq * t)
        s[4] = -wobble_amp * wobble_freq * np.sin(wobble_freq * t)
        s[5] = -descent_rate
        
        return s

# ======================================================================
# 2. EKF
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6)*0.001; self.Q[3:6, 3:6] *= 10.0 # Trust velocity changes
        self.R = np.eye(3)*0.005; self.P = np.eye(6)*0.1
    
    def step(self, z):
        self.x = self.F@self.x; self.P = self.F@self.P@self.F.T + self.Q
        y = z - self.H@self.x; S = self.H@self.P@self.H.T + self.R
        K = self.P@self.H.T@np.linalg.inv(S)
        self.x = self.x + K@y; self.P = (np.eye(6)-K@self.H)@self.P
        return self.x
    
    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        for i in range(steps): tmp = Fp @ tmp; fut[i,:] = tmp
        return fut

# ======================================================================
# 3. AGGRESSIVE DIVE PLANNER
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs
        self.r_obs_safe = r_obs + 0.25
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt

    def request(self, chaser, preds):
        with self.lock: self.req = (chaser, preds)
    def get(self):
        with self.lock: return self.res

    def run(self):
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

    def _generate_guess(self, start, goal):
        ref = np.zeros((6, self.N))
        
        # Check Line of Sight
        vec = goal - start
        dist = np.linalg.norm(vec)
        obs_vec = self.p_obs - start
        proj = np.dot(obs_vec, vec/dist) if dist > 0.01 else 0
        
        # Blocked logic
        blocked = (0 < proj < dist) and (np.linalg.norm(start + proj*(vec/dist) - self.p_obs) < self.r_obs_safe)
        
        if blocked:
            # Go AROUND (Y-axis shift)
            safe_pt = self.p_obs.copy()
            safe_pt[1] -= 1.5 
            
            mid = self.N // 2
            for k in range(self.N):
                if k < mid:
                    al = k/mid
                    ref[0:3, k] = (1-al)*start + al*safe_pt
                else:
                    al = (k-mid)/(self.N-mid-1)
                    ref[0:3, k] = (1-al)*safe_pt + al*goal
        else:
            for k in range(self.N):
                al = k/(self.N-1)
                ref[0:3, k] = (1-al)*start + al*goal
        return ref

    def _solve_scp(self, start, preds):
        if self.prev_sol is None:
            x_ref = self._generate_guess(start[0:3], preds[-1, 0:3])
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = preds[-1]

        for _ in range(3):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack = cp.Variable(self.N, nonneg=True)
            
            dock = np.array([-0.1, 0, 0])
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # Tracking
                w = 0.95**k
                cost += w * 50 * cp.sum_squares(x[0:3, k+1] - (preds[k+1,0:3]+dock))
                cost += 0.01 * cp.sum_squares(u[:,k])

            # Obstacle
            for k in range(1, self.N):
                vec = x_ref[0:3,k] - self.p_obs
                dist = np.linalg.norm(vec)
                n = vec/dist if dist > 0.01 else np.array([0,1,0])
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs_safe - slack[k]]
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += cp.sum(slack) * 1e6
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref

# ======================================================================
# 4. MAIN RUN
# ======================================================================
def run():
    CHASER_START = np.array([-3.0, 0.0, 1.5]) 
    
    # Obstacle Setup
    OBS_POS = np.array([-1.5, 0.0, 1.5]) 
    OBS_RAD = 0.4
    
    target_gen = UnstableHoverTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    ekf.x = target_gen.get_state(0)
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, OBS_POS, OBS_RAD)
    planner.start()
    
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=2,
                     initial_xyzs=np.array([CHASER_START, target_gen.get_state(0)[0:3]]),
                     physics=Physics.PYB, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
                     gui=True)
    PYB = env.getPyBulletClient()
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RAD),
                      p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RAD, rgbaColor=[1,0,0,0.5]),
                      OBS_POS, physicsClientId=PYB)
    safe_r = OBS_RAD + 0.25
    p.addUserDebugLine(OBS_POS+[0,safe_r,0], OBS_POS-[0,safe_r,0], [1,1,0], 2, physicsClientId=PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    curr_traj = None
    last_plan = 0
    
    print(f"{'TIME':<6} | {'CHASER Z':<10} | {'TARGET Z':<10} | {'DIST':<10}")
    print("-" * 50)
    
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
        ekf.step(true_state[0:3] + np.random.normal(0, 0.02, 3))
        
        res = planner.get()
        if res is not None: 
            curr_traj = res
            p.removeAllUserDebugItems(physicsClientId=PYB)
            p.addUserDebugLine(OBS_POS+[0,safe_r,0], OBS_POS-[0,safe_r,0], [1,1,0], 2, physicsClientId=PYB)
            for j in range(HORIZON-1):
                p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

        if sim_t - last_plan > PLAN_INTERVAL:
            preds = ekf.predict_future(HORIZON, DT_PLAN)
            chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
            planner.request(chaser_st, preds)
            last_plan = sim_t
            
            for j in range(HORIZON-1):
                p.addUserDebugLine(preds[j,0:3], preds[j+1,0:3], [0,1,0], 1, physicsClientId=PYB)

        if curr_traj is not None:
            # Aggressive Lead: Target 2 steps ahead to fight wobble lag
            pt = curr_traj[0:3, 2] 
            vt = curr_traj[3:6, 2]
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
        else:
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START, np.zeros(3))
        
        action[1] = np.zeros(4)
        
        if i % 20 == 0:
            c_pos = obs[0][0:3]
            t_pos = obs[1][0:3]
            dist = np.linalg.norm(c_pos - t_pos)
            print(f"{sim_t:05.2f}  | {c_pos[2]:+.2f}      | {t_pos[2]:+.2f}      | {dist:.3f}")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()