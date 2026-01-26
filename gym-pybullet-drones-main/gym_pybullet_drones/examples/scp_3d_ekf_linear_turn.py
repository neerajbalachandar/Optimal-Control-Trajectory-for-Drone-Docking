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
PLAN_INTERVAL = 0.2     # FASTER REACTION: Plan every 0.2s (5Hz)
HORIZON = 15            # Look ahead 1.5s
DT_PLAN = 0.1           
DURATION_SEC = 25.0     # EXTENDED: 25 seconds to see full behavior
TURN_TIME = 4.0         # Turn early at 4s

# ======================================================================
# 1. GROUND TRUTH (SUDDEN TURN TARGET)
# ======================================================================
class SuddenTurnTarget:
    def get_state(self, t):
        s = np.zeros(6)
        speed = 0.5 # Faster target to make it interesting
        
        if t < TURN_TIME:
            # Phase 1: Move +X
            s[0] = 0.0 + speed * t
            s[1] = 0.0
            s[2] = 1.0
            s[3] = speed
            s[4] = 0.0
            s[5] = 0.0
        else:
            # Phase 2: Turn to +Y
            # Freeze X, Start Y
            x_freeze = 0.0 + speed * TURN_TIME
            dt_turn = t - TURN_TIME
            
            s[0] = x_freeze       
            s[1] = 0.0 + speed * dt_turn 
            s[2] = 1.0
            s[3] = 0.0            
            s[4] = speed          
            s[5] = 0.0
            
        return s

# ======================================================================
# 2. FAST ADAPTIVE EKF
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        
        # High Process Noise for velocity = Fast adaptation to turns
        self.Q = np.eye(6) * 0.001 
        self.Q[3:6, 3:6] *= 10.0 
        self.R = np.eye(3) * 0.01 
        self.P = np.eye(6) * 0.1

    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        
        # Velocity Decay: Prevents predicting infinite overshoot
        damping = 0.90 
        
        for i in range(steps):
            tmp[3:6] *= damping 
            tmp = Fp @ tmp
            fut[i,:] = tmp
        return fut

# ======================================================================
# 3. BACKGROUND PLANNER (ASYNC FOR SMOOTHNESS)
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.daemon = True
        self.lock = threading.Lock()
        self.req = None; self.res = None
        self.prev_sol = None
        
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
                    traj = self._plan(data[0], data[1])
                    with self.lock: self.res = traj
                except: pass
            time.sleep(0.01)

    def _plan(self, start, preds):
        # Warm Start
        ref = np.zeros((6, self.N))
        if self.prev_sol is None:
            # Cold
            for k in range(self.N):
                al = k/(self.N-1)
                ref[0:3,k] = (1-al)*start[0:3] + al*preds[-1,0:3]
                ref[1,k] += 1.0 * np.sin(np.pi*al) 
        else:
            # Shift
            ref[:, :-1] = self.prev_sol[:, 1:]
            ref[:, -1] = preds[-1]
            self.prev_sol = ref 

        # SCP Optimization
        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack = cp.Variable(self.N, nonneg=True)
            
            dock = np.array([-0.3, 0, 0])
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # Higher weight on immediate steps to ensure we TURN NOW
                w = 0.8**k
                cost += w * 30 * cp.sum_squares(x[0:3, k+1] - (preds[k+1,0:3]+dock))
                cost += 0.1 * cp.sum_squares(u[:,k])

            # Obstacle
            for k in range(1, self.N):
                vec = ref[0:3,k] - self.p_obs
                dist = np.linalg.norm(vec)
                n = vec/dist if dist > 0.01 else np.array([0,1,0])
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack[k]]
                con += [cp.norm(x[:,k] - ref[:,k]) <= 0.5]

            cost += cp.sum(slack)*10000
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            ref = x.value
            
        self.prev_sol = ref
        return ref

# ======================================================================
# 4. MAIN
# ======================================================================
def run():
    # Start Chaser further back to give it room to accelerate
    CHASER_START = np.array([-1.5, 0.0, 1.0])
    OBS_POS = np.array([-0.5, 0.0, 1.0]) # Move obstacle out of the way for this test
    OBS_RAD = 0.2
    
    target_gen = SuddenTurnTarget()
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
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    curr_traj = None
    last_plan = 0
    
    print(f"{'TIME':<6} | {'CHASER (X,Y)':<18} | {'TARGET (X,Y)':<18} | {'STATUS'}")
    print("-" * 65)
    
    START = time.time()
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        sim_t = i / CTRL_FREQ
        
        # 1. Physics & Sense
        obs, _, _, _, _ = env.step(action)
        true_state = target_gen.get_state(sim_t)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
        
        # 2. EKF
        meas = true_state[0:3] + np.random.normal(0, 0.02, 3)
        ekf.step(meas)
        
        # 3. Planner Interface
        res = planner.get()
        if res is not None: 
            curr_traj = res
            # Draw Plan (Blue)
            p.removeAllUserDebugItems(physicsClientId=PYB)
            # Re-draw obstacle (it gets cleared)
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RAD),
                      p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RAD, rgbaColor=[1,0,0,0.5]),
                      OBS_POS, physicsClientId=PYB)
            
            for j in range(HORIZON-1):
                p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

        if sim_t - last_plan > PLAN_INTERVAL:
            preds = ekf.predict_future(HORIZON, DT_PLAN)
            chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
            planner.request(chaser_st, preds)
            last_plan = sim_t
            
            # Draw Prediction (Green)
            for j in range(HORIZON-1):
                p.addUserDebugLine(preds[j,0:3], preds[j+1,0:3], [0,1,0], 1, physicsClientId=PYB)

        # 4. Control
        if curr_traj is not None:
            # Target the 2nd waypoint (0.1s ahead) for aggressive tracking
            pt = curr_traj[0:3, 1] 
            vt = curr_traj[3:6, 1]
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
        
        action[1] = np.zeros(4)
        
        if i % 20 == 0: # Log every 0.4s
            c_pos = f"{obs[0][0]:.2f},{obs[0][1]:.2f}"
            t_pos = f"{obs[1][0]:.2f},{obs[1][1]:.2f}"
            status = "TURN!" if sim_t >= TURN_TIME else "Linear"
            if sim_t > TURN_TIME + 2.0: status = "Chasing Y"
            print(f"{sim_t:05.2f}  | {c_pos:<18} | {t_pos:<18} | {status}")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()