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
PLAN_INTERVAL = 0.5     # Request a new plan every 0.5 seconds
HORIZON = 15            # Look ahead 1.5s
DT_PLAN = 0.1           

# ======================================================================
# 1. HELPER CLASSES (Target & EKF)
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
# 2. BACKGROUND PLANNER THREAD (The Fix)
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.daemon = True # Kill thread when main program ends
        
        # Shared Data
        self.lock = threading.Lock()
        self.request_data = None # (chaser_state, target_preds)
        self.latest_plan = None  # Result trajectory
        
        # Solver Constants
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt
        self.prev_sol = None

    def trigger_update(self, chaser_state, target_preds):
        """Main thread calls this to request a plan"""
        with self.lock:
            self.request_data = (chaser_state, target_preds)

    def get_plan(self):
        """Main thread calls this to get the latest result"""
        with self.lock:
            return self.latest_plan

    def run(self):
        """This runs in the BACKGROUND, never blocking the simulation"""
        print("[Planner Thread] Started.")
        while True:
            # 1. Check for request
            data = None
            with self.lock:
                if self.request_data:
                    data = self.request_data
                    self.request_data = None # Clear request
            
            if data is None:
                time.sleep(0.01) # Sleep if no work
                continue

            # 2. Solve SCP (This takes time, but doesn't freeze sim)
            chaser_st, preds = data
            try:
                traj = self._solve_scp(chaser_st, preds)
                with self.lock:
                    self.latest_plan = traj
            except Exception as e:
                print(f"Solver Error: {e}")

    def _solve_scp(self, chaser_st, preds):
        # Warm Start Logic
        x_ref = np.zeros((6, self.N))
        if self.prev_sol is None:
            # Cold: Straight line
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*chaser_st[0:3] + al*preds[-1,0:3]
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) 
        else:
            # Warm: Shift previous
            self.prev_sol[:, :-1] = self.prev_sol[:, 1:]
            self.prev_sol[:, -1] = preds[-1]
            x_ref = self.prev_sol

        # SCP Optimization
        for _ in range(2): # Fast iterations
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack = cp.Variable(self.N, nonneg=True)
            
            dock = np.array([-0.3, 0, 0])
            cost = 0
            con = [x[:,0] == chaser_st]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                cost += cp.sum_squares(x[0:3, k+1] - (preds[k+1,0:3]+dock)) * 20
                cost += cp.sum_squares(u[:,k]) * 0.1
            
            # Obstacle
            for k in range(1, self.N):
                vec = x_ref[0:3,k] - self.p_obs
                n = vec / np.linalg.norm(vec) if np.linalg.norm(vec)>0.01 else np.array([0,1,0])
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack[k]]
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]
            
            cost += cp.sum(slack)*10000
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP)
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value
            
        self.prev_sol = x_ref
        return x_ref

# ======================================================================
# 3. MAIN RUN
# ======================================================================
def run():
    # Setup
    CHASER_START = np.array([-2.0, 0.0, 1.0])
    OBS_POS = np.array([-0.5, 0.0, 1.0]); OBS_RAD = 0.4
    
    target_gen = LinearTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    ekf.x = target_gen.get_state(0)
    
    # START BACKGROUND PLANNER
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
    
    # State Vars
    current_traj = None
    last_plan_time = 0
    traj_start_idx = 0 # To track progress along the plan
    
    print(f"{'TIME':<6} | {'CHASER (X,Y,Z)':<22} | {'TARGET (X,Y,Z)':<22} | {'FPS'}")
    print("-" * 65)
    
    START = time.time()
    
    # Run for 15 seconds
    for i in range(int(15.0 * CTRL_FREQ)):
        sim_t = i / CTRL_FREQ
        loop_start = time.time()
        
        # 1. SENSE & PHYSICS (Always smooth)
        obs, _, _, _, _ = env.step(action)
        true_state = target_gen.get_state(sim_t)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
        
        # 2. EKF
        meas = true_state[0:3] + np.random.normal(0, 0.02, 3)
        ekf.step(meas)
        
        # 3. PLANNER INTERACTION
        # A. Check for NEW plan from thread
        new_plan = planner.get_plan()
        if new_plan is not None:
            # We got a new plan! Switch to it.
            current_traj = new_plan
            traj_start_idx = 0 
            # (Optional: Reset planner's output buffer so we don't read same plan twice)
            
            # Visualize Plan (Blue)
            p.removeAllUserDebugItems(physicsClientId=PYB)
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RAD),
                      p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RAD, rgbaColor=[1,0,0,0.5]),
                      OBS_POS, physicsClientId=PYB)
            for j in range(HORIZON-1):
                p.addUserDebugLine(current_traj[0:3,j], current_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

        # B. Trigger NEW calculation if enough time passed
        if sim_t - last_plan_time > PLAN_INTERVAL:
            preds = ekf.predict_future(HORIZON, DT_PLAN)
            chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
            planner.trigger_update(chaser_st, preds)
            last_plan_time = sim_t

        # 4. EXECUTE CONTROL
        if current_traj is not None:
            # We move along the trajectory. 
            # Since trajectory points are 0.1s apart and we run at 48Hz (~0.02s),
            # we need to be careful. Simple approach: Target point 1 (0.1s ahead).
            # The planner re-updates fast enough that we mostly track the start of the curve.
            
            # Robustness: Always target the 1st or 2nd waypoint
            pt = current_traj[0:3, 1] 
            vt = current_traj[3:6, 1]
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
        else:
            # Hover before first plan
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START, np.zeros(3))
            
        action[1] = np.zeros(4)
        
        # 5. LOGGING
        if i % 10 == 0:
            c_str = f"{obs[0][0]:.2f},{obs[0][1]:.2f},{obs[0][2]:.2f}"
            t_str = f"{obs[1][0]:.2f},{obs[1][1]:.2f},{obs[1][2]:.2f}"
            fps = 1.0 / (time.time() - loop_start + 1e-6)
            print(f"{sim_t:05.2f}  | {c_str:<22} | {t_str:<22} | {fps:.1f}")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()