import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync


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
        self.N = N
        self.dt = dt
        self.p_obs = p_obs
        self.r_obs_safe = r_obs + 0.25

        self.daemon = True
        self.lock = threading.Lock()

        self.req = None          # (chaser_state, target_preds)
        self.res = None          # planned trajectory
        self.prev_sol = None     # warm start

        # Discrete double-integrator dynamics
        self.A = np.eye(6)
        self.A[0,3] = dt
        self.A[1,4] = dt
        self.A[2,5] = dt

        self.B = np.zeros((6,3))
        self.B[0,0] = 0.5 * dt**2
        self.B[1,1] = 0.5 * dt**2
        self.B[2,2] = 0.5 * dt**2
        self.B[3,0] = dt
        self.B[4,1] = dt
        self.B[5,2] = dt

    # --------------------------------------------------
    # Public API (USED BY SIMULATION)
    # --------------------------------------------------
    def request(self, chaser_state, target_preds):
        with self.lock:
            self.req = (chaser_state, target_preds)

    def get(self):
        with self.lock:
            return self.res

    # --------------------------------------------------
    # Thread loop
    # --------------------------------------------------
    def run(self):
        while True:
            data = None
            with self.lock:
                if self.req is not None:
                    data = self.req
                    self.req = None

            if data is not None:
                try:
                    traj = self._solve_scp(data[0], data[1])
                    with self.lock:
                        self.res = traj
                except Exception as e:
                    print(f"[SCP ERROR] {e}")

            time.sleep(0.01)

    # --------------------------------------------------
    # Initial guess generation (CRITICAL)
    # --------------------------------------------------
    def _generate_guess(self, start_pos, goal_pos):
        ref = np.zeros((6, self.N))

        vec = goal_pos - start_pos
        dist = np.linalg.norm(vec)

        obs_vec = self.p_obs - start_pos
        proj = np.dot(obs_vec, vec/dist) if dist > 1e-3 else 0.0

        blocked = (
            0.0 < proj < dist and
            np.linalg.norm(start_pos + proj * (vec/dist) - self.p_obs) < self.r_obs_safe
        )

        if blocked:
            safe_pt = self.p_obs.copy()
            safe_pt[1] -= 1.5

            mid = self.N // 2
            for k in range(self.N):
                if k < mid:
                    a = k / mid
                    ref[0:3, k] = (1 - a) * start_pos + a * safe_pt
                else:
                    a = (k - mid) / (self.N - mid - 1)
                    ref[0:3, k] = (1 - a) * safe_pt + a * goal_pos
        else:
            for k in range(self.N):
                a = k / (self.N - 1)
                ref[0:3, k] = (1 - a) * start_pos + a * goal_pos

        return ref

    # --------------------------------------------------
    # SCP SOLVER
    # --------------------------------------------------
    def _solve_scp(self, chaser_state, preds):

        if self.prev_sol is None:
            x_ref = self._generate_guess(
                chaser_state[0:3],
                preds[-1, 0:3]
            )
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1]  = preds[-1]

        for _ in range(3):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N - 1))
            slack = cp.Variable(self.N, nonneg=True)

            dock = np.array([-0.1, 0.0, 0.0])
            cost = 0.0

            con = [x[:,0] == chaser_state]

            for k in range(self.N - 1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]

                w = 0.95 ** k
                cost += w * 50.0 * cp.sum_squares(
                    x[0:3, k+1] - (preds[k+1, 0:3] + dock)
                )
                cost += 0.01 * cp.sum_squares(u[:,k])

            for k in range(1, self.N):
                vec = x_ref[0:3, k] - self.p_obs
                dist = np.linalg.norm(vec)
                n = vec / dist if dist > 1e-3 else np.array([0, 1, 0])

                con += [
                    n @ (x[0:3, k] - self.p_obs)
                    >= self.r_obs_safe - slack[k]
                ]
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += 1e6 * cp.sum(slack)

            prob = cp.Problem(cp.Minimize(cost), con)
            try:
                prob.solve(solver=cp.OSQP)
            except:
                prob.solve(solver=cp.SCS)

            if x.value is None:
                break

            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref
