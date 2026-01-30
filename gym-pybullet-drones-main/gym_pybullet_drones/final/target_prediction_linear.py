import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.final.scp_planner import *


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
    """
    Asynchronous wrapper around plan_scp_docking()

    - No SCP math here
    - Pure scheduling + warm-start logic
    """

    def __init__(
        self,
        N,
        dt,
        p_obs,
        r_obs,
        docking_axis,
        cone_angle_deg
    ):
        super().__init__()
        self.daemon = True

        # Planning params
        self.N = N
        self.dt = dt
        self.p_obs = p_obs
        self.r_obs = r_obs
        self.docking_axis = docking_axis
        self.cone_angle_deg = cone_angle_deg

        # Thread-safe buffers
        self._lock = threading.Lock()
        self._request = None   # (chaser_pos, target_pred_traj)
        self._solution = None  # (N,3)

    # --------------------------------------------------------
    # Public API (used by simulation)
    # --------------------------------------------------------
    def request(self, chaser_state, target_preds):
        """
        chaser_state : (6,)  [pos, vel]
        target_preds : (N,6) predicted target states
        """
        with self._lock:
            self._request = (chaser_state.copy(), target_preds.copy())

    def get(self):
        """Returns latest planned trajectory or None"""
        with self._lock:
            return None if self._solution is None else self._solution.copy()

    # --------------------------------------------------------
    # Thread loop
    # --------------------------------------------------------
    def run(self):
        while True:

            with self._lock:
                data = self._request
                self._request = None

            if data is None:
                time.sleep(0.01)
                continue

            chaser_state, target_preds = data

            try:
                traj = self._solve(chaser_state, target_preds)
                with self._lock:
                    self._solution = traj
            except Exception as e:
                print(f"[AsyncPlanner] SCP failed: {e}")

    # --------------------------------------------------------
    # Core solver wrapper
    # --------------------------------------------------------
    def _solve(self, chaser_state, target_preds):
        """
        Calls the offline SCP planner with a moving target.
        """

        p_start = chaser_state[0:3]
        p_goal  = target_preds[-1, 0:3]   # terminal predicted target pose

        traj = plan_scp_docking(
            p_start=p_start,
            p_goal=p_goal,
            p_obs=self.p_obs,
            r_obs=self.r_obs,
            docking_axis=self.docking_axis,
            cone_angle_deg=self.cone_angle_deg,
            N=self.N,
            dt=self.dt
        )

        # traj shape: (N,3)
        return traj