# ../examples/scp_3d_ekf_unstable_hover.py
# Linear, Parabolic turns, Spiral Fall Target cases

import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync


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
    
