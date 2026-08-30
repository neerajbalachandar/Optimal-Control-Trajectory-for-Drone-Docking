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
PLAN_INTERVAL = 0.05      # Faster replanning (20 Hz)
HORIZON = 160             # Shorter horizon for faster online solve/update
DT_PLAN = 1.0 / 48.0      # Paper: dt = 1/48 s
MAX_SCP_ITERS = 10
DURATION_SEC = 35.0

# --- DOCKING & SAFETY ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  
USE_REACTIVE_BACKOFF = False
# Paper-style convex limits to keep SCP commands physically admissible.
U_MAX = 3.5           # m/s^2, translational SCP input bound
Z_MIN = 0.05          # m
Z_MAX = 2.5           # m
V_REF_MAX = 1.2       # m/s, clip velocity references sent to PID
U_FF_GAIN = 0.06      # feed-forward blending gain for u*
TRAJ_LOOKAHEAD_IDX = 2
PRE_DOCK_OFFSET_DIST = 0.25
PRE_DOCK_SWITCH_DIST = 0.16
PRE_DOCK_MIN_ABOVE = 0.15
SLOW_DOCK_START_DIST = 0.25
DOCK_DESCENT_RATE = 0.10       # m/s
V_REF_MAX_TERMINAL = 0.45      # m/s
DOCK_HOLD_STEPS = 4            # consecutive control steps for terminal success
DOCK_REL_POS_TOL = 0.10        # docking-stage completion tolerance
DOCK_REL_VEL_TOL = 0.18        # paper-like terminal velocity tolerance
CONE_EVAL_MARGIN = 0.005
EARLY_STOP_ENABLE = True
EARLY_STOP_REL_POS_TOL = 0.18  # "good enough" capture radius (no physical docking required)
EARLY_STOP_REL_VEL_TOL = 0.28
EARLY_STOP_HOLD_STEPS = 8
EARLY_STOP_DCOL_ALPHA = 1.12   # DCOL-style safety gate wrt obstacle before stopping

# Debug-only actuation mode:
# Interpret SCP u* as desired world acceleration and inject equivalent world force.
# This bypasses proper attitude/RPM mapping and is only for diagnosing planner vs controller mismatch.
USE_SCP_DIRECT_FORCE_DEBUG = False
DIRECT_FORCE_SCALE = 1.0
DIRECT_FORCE_ALPHA = 0.35
USE_TRUE_TARGET_PREDICTION = True
ENABLE_TERMINAL_REPLANNING = False
TERM_REL_KP = 2.2
TERM_REL_KD = 1.6
RUNTIME_OBS_MARGIN = 0.02
RUNTIME_CONE_LONG_MARGIN = 0.0
RUNTIME_MAX_REL_LATERAL_SPEED = 0.35
TERM_CONE_RECOVERY_KP = 4.0
TERM_CONE_RECOVERY_LONG_GAIN = 0.0
TERM_CONE_RECOVERY_MAX_UP = 0.05
TERMINAL_WIND_SCALE = 0.0
DCOL_OBS_HARD_ALPHA = 1.08     # alpha < this => force obstacle-escape guard
DCOL_OBS_ESCAPE_ALPHA = 1.14   # shell to project command back to
DCOL_ESCAPE_KP = 2.5
DCOL_ESCAPE_KD = 1.2

# SCP objective and constraint weights (explicitly named for diagnostics and reporting).
W_TRACK_POS = 95.0
W_TRACK_VEL = 12.0
W_TRACK_TERM_POS = 2400.0
W_TRACK_TERM_VEL = 320.0
W_CONTROL = 0.1
W_CONTROL_DERIV = 0.05
W_SLACK_OBS = 1e6
W_SLACK_CONE = 5e4
W_SLACK_MONO = 2e4
W_SLACK_TERM = 5e5
TRUST_REGION_RADIUS = 0.5
TERM_POS_TOL = 0.05
TERM_VEL_TOL = 0.1
EXTRA_OBS_MARGIN = 0.05
CONE_ACTIVE_FRAC = 0.70

# Wind
WIND_NOMINAL  = np.array([0.03, -0.01, -0.01])
WIND_GUST_AMP = 0.02

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. LINEAR TARGET (Ground Truth)
# ======================================================================
class LinearTarget:
    def get_state(self, t):
        s = np.zeros(6)
        # Tuned linear case: slower target for more reliable docking tests
        s[0] = 0.5 + 0.12 * t   # X
        s[1] = 0.0            # Y
        s[2] = 1.0            # Z
        
        # Velocity
        s[3] = 0.12
        s[4] = 0.0
        s[5] = 0.0
        return s

# ======================================================================
# 2. TARGET EKF
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6)*0.01; self.R = np.eye(3)*0.01
        self.P = np.eye(6)*0.1
        self.last_z = None
        self.vel_lpf = np.zeros(3)
        self.vel_alpha = 0.55
        self.vel_blend = 0.60
        self.max_pred_speed = 2.0

    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ self.H) @ self.P
        if self.last_z is not None and self.dt > 1e-9:
            z_vel = (z - self.last_z) / self.dt
            self.vel_lpf = (1.0 - self.vel_alpha) * self.vel_lpf + self.vel_alpha * z_vel
            v_meas = np.clip(self.vel_lpf, -self.max_pred_speed, self.max_pred_speed)
            self.x[3:6] = (1.0 - self.vel_blend) * self.x[3:6] + self.vel_blend * v_meas
        self.last_z = z.copy()
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        for i in range(steps): tmp = Fp @ tmp; fut[i,:] = tmp
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
    # Initialize static storage for line IDs if it doesn't exist
    if not hasattr(draw_dynamic_cone, "ids"):
        draw_dynamic_cone.ids = []

    # Axis [0,0,-1] -> Approach [0,0,1]
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis 
    
    length = 1.0 # Short cone to visualize immediate docking area
    theta = np.deg2rad(angle_deg)
    
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    line_idx = 0
    # Draw Radial Lines (30 lines)
    for phi in np.linspace(0, 2*np.pi, 30):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        
        if len(draw_dynamic_cone.ids) <= line_idx:
            # Create new line (lifeTime=0 makes it persistent)
            uid = p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, physicsClientId=client)
            draw_dynamic_cone.ids.append(uid)
        else:
            # Update existing line
            p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
        
        line_idx += 1

    # Draw Center Axis Line
    if len(draw_dynamic_cone.ids) <= line_idx:
        uid = p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, physicsClientId=client)
        draw_dynamic_cone.ids.append(uid)
    else:
        p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)

def draw_planned_trajectory(traj, client):
    if traj is None or traj.shape[1] < 2:
        return
    if not hasattr(draw_planned_trajectory, "ids"):
        draw_planned_trajectory.ids = []
    line_idx = 0
    for j in range(traj.shape[1]-1):
        p0 = traj[0:3, j]
        p1 = traj[0:3, j+1]
        if len(draw_planned_trajectory.ids) <= line_idx:
            uid = p.addUserDebugLine(p0, p1, [0, 0, 1], 3, lifeTime=0, physicsClientId=client)
            draw_planned_trajectory.ids.append(uid)
        else:
            p.addUserDebugLine(
                p0, p1, [0, 0, 1], 3, lifeTime=0,
                replaceItemUniqueId=draw_planned_trajectory.ids[line_idx],
                physicsClientId=client
            )
        line_idx += 1

def project_to_obstacle_clearance(p_cmd, p_obs, min_clearance):
    vec = p_cmd - p_obs
    d = np.linalg.norm(vec)
    if d >= min_clearance:
        return p_cmd
    if d < 1e-9:
        vec = np.array([1.0, 0.0, 0.0])
        d = 1.0
    return p_obs + (min_clearance / d) * vec

def project_to_docking_cone(p_cmd, p_target, docking_axis, cone_angle_deg, long_margin=0.0):
    n_app = docking_axis / (np.linalg.norm(docking_axis) + 1e-9)
    approach_dir = -n_app
    rel = p_cmd - p_target
    long = np.dot(rel, approach_dir)
    perp = rel - long * approach_dir
    long = max(long, long_margin)
    max_perp = np.tan(np.deg2rad(cone_angle_deg)) * long
    perp_n = np.linalg.norm(perp)
    if perp_n > max_perp:
        if perp_n < 1e-9:
            perp = np.zeros(3)
        else:
            perp = perp * (max_perp / perp_n)
    return p_target + long * approach_dir + perp

def enforce_runtime_constraints(p_cmd, p_target, p_obs, min_clearance, docking_axis, cone_angle_deg, enforce_cone):
    p_safe = project_to_obstacle_clearance(p_cmd, p_obs, min_clearance)
    if enforce_cone:
        p_safe = project_to_docking_cone(
            p_safe,
            p_target,
            docking_axis,
            cone_angle_deg,
            long_margin=RUNTIME_CONE_LONG_MARGIN
        )
    return p_safe

def enforce_dcol_obstacle_guard(p_cmd, v_cmd, p_chaser, v_chaser, p_obs, r_safe, r_obs):
    # DCOL-style collision metric: alpha > 1 means separated, alpha < 1 means interpenetrating.
    alpha_obs = solve_dcol_scaling(p_chaser, r_safe, p_obs, r_obs)
    if alpha_obs >= DCOL_OBS_HARD_ALPHA:
        return p_cmd, v_cmd, False, alpha_obs

    vec = p_chaser - p_obs
    d = np.linalg.norm(vec)
    if d < 1e-9:
        vec = np.array([1.0, 0.0, 0.0])
        d = 1.0
    unit = vec / d
    safe_shell = p_obs + DCOL_OBS_ESCAPE_ALPHA * (r_safe + r_obs) * unit
    p_safe = safe_shell
    v_safe = DCOL_ESCAPE_KP * (p_safe - p_chaser) - DCOL_ESCAPE_KD * v_chaser
    return p_safe, v_safe, True, alpha_obs

    
     

# ======================================================================
# 4. ASYNC PLANNER
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, r_safe, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.r_safe = r_safe; self.obs_clearance = r_obs + r_safe + EXTRA_OBS_MARGIN
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None; self.prev_u = None
        self._last_error_print_t = 0.0
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt
        try:
            self.available_solvers = set(cp.installed_solvers())
        except Exception:
            self.available_solvers = set()

    def request(self, chaser, preds, dock_offset=None, enforce_cone=True):
        if dock_offset is None:
            dock_offset = np.zeros(3)
        with self.lock:
            self.req = (chaser, preds, np.array(dock_offset, dtype=float), bool(enforce_cone))
    def get(self):
        with self.lock:
            out = self.res
            self.res = None
            return out

    def reset_warm_start(self):
        with self.lock:
            self.prev_sol = None
            self.prev_u = None

    def _solve_problem(self, problem):
        # Prefer Clarabel for conic SCP subproblems; fall back deterministically.
        for solver_name in ("CLARABEL", "SCS", "ECOS"):
            if solver_name not in self.available_solvers:
                continue
            try:
                if solver_name == "CLARABEL":
                    problem.solve(solver=solver_name, warm_start=True, verbose=False, max_iter=80)
                elif solver_name == "SCS":
                    problem.solve(solver=solver_name, warm_start=True, verbose=False, max_iters=800, eps=1e-3)
                else:
                    problem.solve(solver=solver_name, warm_start=True, verbose=False)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    return solver_name
            except Exception:
                continue
        problem.solve(warm_start=True, verbose=False)
        return str(problem.solver_stats.solver_name) if problem.solver_stats is not None else "DEFAULT"

    def _build_scp_subproblem(self, start, preds, x_ref, cos_theta, n_app, dock_offset, enforce_cone):
        x = cp.Variable((6, self.N))
        u = cp.Variable((3, self.N-1))
        slack_obs = cp.Variable(self.N, nonneg=True)
        slack_cone = cp.Variable(self.N, nonneg=True)
        slack_mono = cp.Variable(self.N-1, nonneg=True)
        slack_term_p = cp.Variable(nonneg=True)
        slack_term_v = cp.Variable(nonneg=True)

        cost_track = 0
        cost_track_vel = 0
        cost_terminal_pos = 0
        cost_terminal_vel = 0
        cost_control = 0
        cost_control_deriv = 0
        con = [x[:,0] == start]
        cone_start_idx = int(max(1, CONE_ACTIVE_FRAC * (self.N - 1)))
        dist_long_expr = [None] * self.N

        for k in range(self.N-1):
            con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
            con += [cp.norm_inf(u[:,k]) <= U_MAX]
            con += [x[2, k+1] >= Z_MIN, x[2, k+1] <= Z_MAX]

            target_k = preds[k+1, 0:3] + dock_offset
            cost_track += cp.sum_squares(x[0:3, k+1] - target_k)
            cost_track_vel += cp.sum_squares(x[3:6, k+1] - preds[k+1, 3:6])
            cost_control += cp.sum_squares(u[:,k])
            if k < self.N - 2:
                cost_control_deriv += cp.sum_squares(u[:,k+1] - u[:,k])

        for k in range(1, self.N):
            vec = x_ref[0:3, k] - self.p_obs
            n = vec / (np.linalg.norm(vec) + 1e-4)
            con += [n @ (x[0:3, k] - self.p_obs) >= self.obs_clearance - slack_obs[k]]

            if enforce_cone and k >= cone_start_idx:
                p_rel = x[0:3, k] - preds[k, 0:3]
                dist_long = -n_app @ p_rel
                dist_long_expr[k] = dist_long
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]

            con += [cp.norm(x[:, k] - x_ref[:, k]) <= TRUST_REGION_RADIUS]

        if enforce_cone:
            for k in range(cone_start_idx, self.N - 1):
                if dist_long_expr[k] is None or dist_long_expr[k+1] is None:
                    continue
                # Monotone approach in terminal phase: do not move away along docking axis.
                con += [dist_long_expr[k+1] <= dist_long_expr[k] + slack_mono[k]]

        # Terminal capture constraints (softened for feasibility).
        terminal_target = preds[-1, 0:3] + dock_offset
        con += [cp.norm(x[0:3, -1] - terminal_target) <= TERM_POS_TOL + slack_term_p]
        con += [cp.norm(x[3:6, -1] - preds[-1, 3:6]) <= TERM_VEL_TOL + slack_term_v]
        cost_terminal_pos += cp.sum_squares(x[0:3, -1] - terminal_target)
        cost_terminal_vel += cp.sum_squares(x[3:6, -1] - preds[-1, 3:6])

        cost = (
            W_TRACK_POS * cost_track +
            W_TRACK_VEL * cost_track_vel +
            W_TRACK_TERM_POS * cost_terminal_pos +
            W_TRACK_TERM_VEL * cost_terminal_vel +
            W_CONTROL * cost_control +
            W_CONTROL_DERIV * cost_control_deriv +
            W_SLACK_OBS * cp.sum(slack_obs) +
            W_SLACK_CONE * cp.sum(slack_cone) +
            (W_SLACK_MONO * cp.sum(slack_mono) if enforce_cone else 0) +
            W_SLACK_TERM * (slack_term_p + slack_term_v)
        )
        problem = cp.Problem(cp.Minimize(cost), con)
        return problem, x, u

    def _compute_plan_diagnostics(self, x_sol, u_sol, preds, dock_offset, enforce_cone):
        targets = preds[:, 0:3] + dock_offset[None, :]
        pos_term_err = float(np.linalg.norm(x_sol[0:3, -1] - targets[-1]))
        vel_term_err = float(np.linalg.norm(x_sol[3:6, -1] - preds[-1, 3:6]))
        rel_dist = np.linalg.norm((x_sol[0:3, :].T - targets), axis=1)
        start_target_dist = float(rel_dist[0])
        end_target_dist = float(rel_dist[-1])
        min_target_dist = float(np.min(rel_dist))
        obs_margins = np.linalg.norm((x_sol[0:3, :].T - self.p_obs), axis=1) - self.obs_clearance
        min_obs_margin = float(np.min(obs_margins))
        if u_sol is None:
            max_u_inf = float("nan")
        else:
            max_u_inf = float(np.max(np.linalg.norm(u_sol.T, ord=np.inf, axis=1)))
        n_app = self.axis / (np.linalg.norm(self.axis) + 1e-9)
        p_rel_true = x_sol[0:3, :].T - preds[:, 0:3]
        d_long = -p_rel_true @ n_app
        if enforce_cone:
            cone_lhs = np.linalg.norm(p_rel_true, axis=1) * np.cos(np.deg2rad(self.cone_angle)) - d_long
            max_cone_violation = float(max(0.0, np.max(cone_lhs)))
            max_dlong_increase = float(max(0.0, np.max(d_long[1:] - d_long[:-1])))
            min_dlong = float(np.min(d_long))
        else:
            max_cone_violation = float("nan")
            max_dlong_increase = float("nan")
            min_dlong = float("nan")
        return {
            "term_pos_err": pos_term_err,
            "term_vel_err": vel_term_err,
            "start_target_dist": start_target_dist,
            "end_target_dist": end_target_dist,
            "min_target_dist": min_target_dist,
            "min_obs_margin": min_obs_margin,
            "max_u_inf": max_u_inf,
            "max_cone_violation": max_cone_violation,
            "max_dlong_increase": max_dlong_increase,
            "min_dlong": min_dlong,
            "cone_enforced": bool(enforce_cone),
        }

    def run(self):
        print("[Planner] Thread Started.")
        while True:
            data = None
            with self.lock:
                if self.req: data = self.req; self.req = None
            if data:
                try:
                    traj = self._solve_scp(data[0], data[1], data[2], data[3])
                    with self.lock: self.res = traj
                except Exception as e:
                    now = time.time()
                    if now - self._last_error_print_t > 1.0:
                        print(f"[Planner][WARN] SCP solve error: {e}")
                        self._last_error_print_t = now
            time.sleep(0.01)

    def _solve_scp(self, start, preds, dock_offset, enforce_cone):
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*start[0:3] + al*(preds[-1,0:3] + dock_offset)
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) # Arc for start
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[0:3, -1] = preds[-1, 0:3] + dock_offset
            x_ref[3:6, -1] = preds[-1, 3:6]

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        u_ref = self.prev_u
        used_solver = None
        plan_diag = None
        solve_t0 = time.time()
        rel_tol = 1e-3
        time_budget_sec = 0.10
        for _ in range(MAX_SCP_ITERS):
            x_prev = x_ref.copy()
            prob, x, u = self._build_scp_subproblem(start, preds, x_ref, cos_theta, n_app, dock_offset, enforce_cone)
            used_solver = self._solve_problem(prob)
            
            if x.value is None: break
            x_ref = x.value
            u_ref = u.value
            plan_diag = self._compute_plan_diagnostics(x_ref, u_ref, preds, dock_offset, enforce_cone)
            rel_change = np.linalg.norm(x_ref - x_prev) / (np.linalg.norm(x_prev) + 1e-6)
            if rel_change < rel_tol:
                break
            if (time.time() - solve_t0) > time_budget_sec:
                break

        self.prev_sol = x_ref
        self.prev_u = u_ref
        return {
            "x": x_ref,
            "u": u_ref,
            "solver": used_solver,
            "diag": plan_diag,
            "dock_offset": dock_offset.copy(),
            "enforce_cone": bool(enforce_cone),
        }


def plot_performance_legacy(history, docking_threshold=0.1):

    t = np.array(history['t'])
    p_c = np.array(history['p_c'])
    p_t = np.array(history['p_t'])
    v_c = np.array(history['v_c'])
    act = np.array(history['action']) # Raw RPMs

    # -------------------------------------------------
    # Distance & Error
    # -------------------------------------------------
    err_vec = p_c - p_t
    dist = np.linalg.norm(err_vec, axis=1)

    # -------------------------------------------------
    # Control Effort (Non-Dimensional Thrust: T / T_hover)
    # -------------------------------------------------
    start_idx = min(len(t)-1, 50)
    hover_rpm = np.mean(act[start_idx:], axis=0)
    
    # Physics: Thrust is proportional to RPM squared
    thrust_proxy = np.sum(act**2, axis=1)
    hover_thrust_proxy = np.sum(hover_rpm**2) + 1e-6 # Add epsilon to avoid div-by-zero
    
    # Dimensionless Ratio (T / T_hover)
    nd_thrust = thrust_proxy / hover_thrust_proxy

    # -------------------------------------------------
    # Compute Quantitative Metrics
    # -------------------------------------------------
    d0 = dist[0]
    df = dist[-1]
    emax = np.max(dist)
    
    # Filter out absurd spikes for the print metric
    valid_thrust = nd_thrust[start_idx:]
    t_max_nd = np.percentile(valid_thrust, 99.5) if len(valid_thrust) > 0 else 1.0

    idx = np.where(dist < docking_threshold)[0]
    td = t[idx[0]] if len(idx) > 0 else None
    

    # Raw deviation
    act_dev = act - hover_rpm
    control_norm = np.linalg.norm(act_dev, axis=1)
    
    # Normalize by the magnitude of the hover state to make it dimensionless
    hover_norm = np.linalg.norm(hover_rpm) + 1e-6 # Add epsilon to avoid div by zero
    normalized_control = control_norm / hover_norm

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Initial Distance d0        : {d0:.4f} m")
    print(f"Final Distance df          : {df:.4f} m")
    print(f"Maximum Tracking Error     : {emax:.4f} m")
    print(f"Peak Thrust Ratio (99.5%)  : {t_max_nd:.4f}")
    print(f"Docking Time td            : {td:.4f} s" if td else "Docking Time td: Not reached")
    print("--------------------------------\n")

    # -------------------------------------------------
    # --- PLOTS ---
    # -------------------------------------------------

    # 1. 3D TRAJECTORY
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
    ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
    ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
    ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # 2. POSITION EVOLUTION
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,0], 'b-', label='X')
    plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,1], 'r-', label='Y')
    plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,2], 'k-', label='Z')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()

    # 3. TRACKING ERROR (Scaled for Paper)
    fig3 = plt.figure(figsize=(4.5, 3.4), dpi=300)
    plt.plot(t, err_vec[:,0], color='#1f77b4', linewidth=2.0, label='Err X')
    plt.plot(t, err_vec[:,1], color='#d62728', linewidth=2.0, label='Err Y')
    plt.plot(t, err_vec[:,2], color='black', linewidth=2.0, label='Err Z')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Tracking Error (m)", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(bottom=0.18) 

    # 4. VELOCITY
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(t, v_c[:,0], label='Vx')
    plt.plot(t, v_c[:,1], label='Vy')
    plt.plot(t, v_c[:,2], label='Vz')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    # 5. CONTROL EFFORT (Non-Dimensional Thrust)
    fig5 = plt.figure(figsize=(4.2, 3.0), dpi=300)

    # Large smoothing window for clean trend
    window = 35
    thrust_smooth = np.convolve(
        nd_thrust,
        np.ones(window)/window,
        mode='same'
    )

    # Raw (noisy, thin)
    plt.plot(t, nd_thrust,
            color='#1f77b4',
            linewidth=1.0,
            label='Raw Thrust')

    # Clean moving average (smooth curve)
    plt.plot(t, thrust_smooth,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Moving Avg')
            
    # Add a baseline for 1.0 (Hover Thrust)
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Hover State')

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Norm. Thrust ($T / T_{hover}$)", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    # 6. DOCKING CONE CONSTRAINT (Terminal Zoom)
    fig6 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    rel_vec = p_c - p_t
    rel_norm = np.linalg.norm(rel_vec, axis=1)
    a_hat = np.array([0, 0, 1])
    cos_theta = np.dot(rel_vec, a_hat) / (rel_norm + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_cone = np.deg2rad(30)

    t_end = t[-1]
    mask = t >= (t_end - 3.0)

    plt.plot(t[mask], np.rad2deg(theta[mask]), color='#1f77b4', linewidth=2.2, label=r'$\theta(t)$')
    plt.axhline(np.rad2deg(theta_cone), color='black', linestyle='--', linewidth=1.8, label=r'$\theta_{cone}$')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Cone Angle (deg)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()

    
    
    fig7 = plt.figure(figsize=(4.2, 3.0), dpi=300)

    # Large smoothing window for clean trend
    window = 35
    control_smooth = np.convolve(
        normalized_control,
        np.ones(window)/window,
        mode='same'
    )

    # Raw (noisy, thin)
    plt.plot(t, normalized_control,
            color='#1f77b4',
            linewidth=1.0,
            label='Raw')

    # Clean moving average (smooth curve)
    plt.plot(t, control_smooth,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Moving Avg')

    plt.xlabel("Time (s)", fontsize=12)
    # Updated Y-label to reflect dimensionless ratio
    plt.ylabel(r"Normalized $||\Delta u||$", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')

    plt.tight_layout()
    
    plt.show()
    

    return d0, df, emax, t_max_nd, td

def plot_performance(history, docking_threshold=0.15):

    t = np.array(history['t'])
    p_c = np.array(history['p_c'])
    p_t = np.array(history['p_t'])
    v_c = np.array(history['v_c'])
    act = np.array(history['action']) # Raw RPMs

    # -------------------------------------------------
    # Recompute Final Metrics on FULL 20s Data
    # -------------------------------------------------
    err_vec = p_c - p_t
    dist = np.linalg.norm(err_vec, axis=1)

    start_idx = min(len(t)-1, 50)
    hover_rpm = np.mean(act[start_idx:], axis=0)
    
    # Thrust proxy (T / T_hover)
    thrust_proxy = np.sum(act**2, axis=1)
    hover_thrust_proxy = np.sum(hover_rpm**2) + 1e-6
    nd_thrust = thrust_proxy / hover_thrust_proxy

    # Normalized Delta u
    act_dev = act - hover_rpm
    control_norm = np.linalg.norm(act_dev, axis=1)
    hover_norm = np.linalg.norm(hover_rpm) + 1e-6 
    normalized_control = control_norm / hover_norm

    # -------------------------------------------------
    # Compute Quantitative Metrics
    # -------------------------------------------------
    d0 = dist[0]
    df = dist[-1]
    emax = np.max(dist)
    
    # Find exact docking time td (but DO NOT truncate the plot data)
    idx = np.where(dist < docking_threshold)[0]
    td = t[idx[0]] if len(idx) > 0 else None
    
    valid_thrust = nd_thrust[start_idx:]
    t_max_nd = np.percentile(valid_thrust, 99.5) if len(valid_thrust) > 0 else 1.0

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Initial Distance d0        : {d0:.4f} m")
    print(f"Final Distance df          : {df:.4f} m")
    print(f"Maximum Tracking Error     : {emax:.4f} m")
    print(f"Peak Thrust Ratio (99.5%)  : {t_max_nd:.4f}")
    print(f"Docking Time td            : {td:.4f} s" if td else "Docking Time td: Not reached")
    print("--------------------------------\n")

    # Dynamic smoothing window based on data length
    window = min(35, max(1, len(t) // 5))

    # -------------------------------------------------
    # --- PLOTS ---
    # -------------------------------------------------

    # 1. 3D TRAJECTORY
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
    ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
    ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
    ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # 2. POSITION EVOLUTION
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(t, p_t[:,0], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,0], 'b-', label='X')
    plt.plot(t, p_t[:,1], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,1], 'r-', label='Y')
    plt.plot(t, p_t[:,2], 'g--', alpha=0.4)
    plt.plot(t, p_c[:,2], 'k-', label='Z')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()

    # 3. TRACKING ERROR (Scaled for Paper)
    fig3 = plt.figure(figsize=(4.5, 3.4), dpi=300)
    plt.plot(t, err_vec[:,0], color='#1f77b4', linewidth=2.0, label='Err X')
    plt.plot(t, err_vec[:,1], color='#d62728', linewidth=2.0, label='Err Y')
    plt.plot(t, err_vec[:,2], color='black', linewidth=2.0, label='Err Z')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Tracking Error (m)", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(bottom=0.18) 

    # 4. VELOCITY
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(t, v_c[:,0], label='Vx')
    plt.plot(t, v_c[:,1], label='Vy')
    plt.plot(t, v_c[:,2], label='Vz')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    # 5. CONTROL EFFORT (Non-Dimensional Thrust)
    fig5 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    thrust_smooth = np.convolve(nd_thrust, np.ones(window)/window, mode='same')
    plt.plot(t, nd_thrust, color='#1f77b4', linewidth=1.0, label='Raw Thrust')
    plt.plot(t, thrust_smooth, color='black', linestyle='--', linewidth=2.5, label='Moving Avg')
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Hover State')
    if len(t) > start_idx:
        p_min, p_max = np.percentile(valid_thrust, 1), np.percentile(valid_thrust, 99)
        margin = max(0.05, (p_max - p_min) * 0.2)
        plt.ylim(max(0, p_min - margin), p_max + margin)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Norm. Thrust ($T / T_{hover}$)", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    # 6. DOCKING CONE CONSTRAINT (Terminal Zoom)
    fig6 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    rel_vec = p_c - p_t
    rel_norm = np.linalg.norm(rel_vec, axis=1)
    a_hat = np.array([0, 0, 1])
    cos_theta = np.dot(rel_vec, a_hat) / (rel_norm + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_cone = np.deg2rad(30)
    
    # To compare 20s fairly, we'll plot the entire timeline instead of zooming the last 3s
    plt.plot(t, np.rad2deg(theta), color='#1f77b4', linewidth=2.2, label=r'$\theta(t)$')
    plt.axhline(np.rad2deg(theta_cone), color='black', linestyle='--', linewidth=1.8, label=r'$\theta_{cone}$')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Cone Angle (deg)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()

    # 7. DELTA U NORMALIZED
    fig7 = plt.figure(figsize=(4.2, 3.0), dpi=300)
    control_smooth = np.convolve(normalized_control, np.ones(window)/window, mode='same')
    plt.plot(t, normalized_control, color='#1f77b4', linewidth=1.0, label='Raw')
    plt.plot(t, control_smooth, color='black', linestyle='--', linewidth=2.5, label='Moving Avg')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"Normalized $||\Delta u||$", fontsize=12) 
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(fontsize=9, frameon=False, loc='best')
    plt.tight_layout()
    
    plt.show()

    return d0, df, emax, t_max_nd, td





    
    
# ======================================================================
# MAIN EXECUTION (LINEAR TARGET)
# ======================================================================
def run():
    # --- FAIR COMPARISON SETUP ---
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION_SEC = 35.0
    
    CHASER_START = np.array([-2.5, 0.0, 1.5])
    P_OBS        = np.array([-1.0, 0.0, 1.25]) 
    R_OBS        = 0.4
    
    target_gen = LinearTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    init_s = target_gen.get_state(0)
    ekf.x = init_s # Init perfect
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, P_OBS, R_OBS, SAFETY_R, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, 
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, init_s[0:3]]),
        physics=Physics.PYB_DW, # Downwash
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
    hover_rpm_cmd = np.ones(4) * env.HOVER_RPM
    direct_u_lpf = np.zeros(3)
    
    state = STATE_TRACKING
    curr_plan = None
    plan_step_idx = 0
    last_solver_name = None
    last_pred_diag_t = -1.0
    last_cone_warn_t = -1.0
    last_dcol_warn_t = -1.0
    # Trigger first planning request immediately.
    last_plan = -PLAN_INTERVAL
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    # Two-phase docking: first stabilize above target, then cone-constrained terminal docking.
    docking_phase = 0
    DOCK_OFFSET = -DOCKING_AXIS * PRE_DOCK_OFFSET_DIST
    
    frozen = False
    freeze_pos_c = None
    freeze_pos_t = None
    dock_standoff = PRE_DOCK_OFFSET_DIST
    dock_hold_counter = 0
    early_stop_counter = 0
    terminal_lock = False
    
    history = {
        't': [], 
        'p_c': [], 'p_t': [], 
        'v_c': [], 'action': []
    }
    
    print("[SIM] Running Linear Target SCP Docking...")
    START = time.time()

    def get_plan_predictions(sim_t_now):
        if USE_TRUE_TARGET_PREDICTION:
            preds = np.zeros((HORIZON, 6))
            for k in range(HORIZON):
                t_k = sim_t_now + (k + 1) * DT_PLAN
                preds[k, :] = target_gen.get_state(t_k)
            return preds
        return ekf.predict_future(HORIZON, DT_PLAN)
    
    # Draw trajectory preview
    for t in np.arange(0, DURATION_SEC, 0.1):
        p.addUserDebugLine(target_gen.get_state(t)[0:3], target_gen.get_state(t+0.1)[0:3], [0.5, 0, 0.5], 2, physicsClientId=PYB)

    # Bootstrap: compute the very first SCP plan before allowing the chaser to move.
    init_chaser_st = np.hstack([CHASER_START, np.zeros(3)])
    init_preds = get_plan_predictions(0.0)
    planner.request(
        init_chaser_st,
        init_preds,
        dock_offset=DOCK_OFFSET,
        enforce_cone=(docking_phase == 1),
    )
    bootstrap_deadline = time.time() + 5.0
    while time.time() < bootstrap_deadline:
        init_res = planner.get()
        if init_res is not None:
            curr_plan = init_res
            plan_step_idx = 0
            if curr_plan.get("solver") is not None:
                last_solver_name = curr_plan.get("solver")
                print(f"[Planner] Active solver: {last_solver_name}")
            if curr_plan.get("diag") is not None:
                d = curr_plan["diag"]
                print(
                    "[Planner] Init diag | "
                    f"term_pos={d['term_pos_err']:.3f}, term_vel={d['term_vel_err']:.3f}, "
                    f"d_start={d['start_target_dist']:.3f}, d_end={d['end_target_dist']:.3f}, "
                    f"min_obs_margin={d['min_obs_margin']:.3f}, max_u_inf={d['max_u_inf']:.3f}, "
                    f"cone={d['cone_enforced']}, cone_violation={d['max_cone_violation']:.4f}, "
                    f"max_dlong_inc={d['max_dlong_increase']:.4f}"
                )
            draw_planned_trajectory(curr_plan["x"], PYB)
            print("[Planner] Initial trajectory ready.")
            break
        env.render()
        time.sleep(0.01)
    if curr_plan is None:
        print("[Planner][WARN] Initial trajectory not ready; holding hover until first plan arrives.")
    
    paused = False   # <-- ADD THIS
    
            
    try:
            for i in range(int(DURATION_SEC * CTRL_FREQ)):
                sim_t = i / CTRL_FREQ
                
                # 0. CHECK WINDOW STATUS
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
                    
                # 1. FREEZE LOGIC
                if frozen:
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], freeze_pos_c, [0,0,0,1], physicsClientId=PYB)
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[1], freeze_pos_t, [0,0,0,1], physicsClientId=PYB)
                    env.render()
                    sync(i, START, env.CTRL_TIMESTEP)
                    continue
                    
                # 2. STEP PHYSICS
                obs, _, _, _, _ = env.step(action)
                true_state = target_gen.get_state(sim_t)
                
                p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
                
                # Visuals Update
                p_chaser = obs[0][0:3]
                p_target = true_state[0:3].copy()
                update_hull(hull_c, p_chaser, PYB)
                update_hull(hull_t, p_target, PYB)
                
                # Wind Disturbance (disabled in direct-force debug mode to isolate actuation effects)
                if USE_SCP_DIRECT_FORCE_DEBUG:
                    wind = np.zeros(3)
                else:
                    wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
                    if docking_phase == 1:
                        wind = TERMINAL_WIND_SCALE * wind
                p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
                p.addUserDebugLine(p_chaser, p_chaser + wind*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
                draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

                # EKF Step
                ekf.step(true_state[0:3] + np.random.normal(0, 0.01, 3))
                if sim_t - last_pred_diag_t >= 1.0:
                    pred_err = np.linalg.norm(ekf.x[0:3] - true_state[0:3])
                    print(f"[Predictor] t={sim_t:.2f}s | EKF pos err={pred_err:.3f} m")
                    last_pred_diag_t = sim_t
                
                # Safety Check
                alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, P_OBS, R_OBS)
                n_app = DOCKING_AXIS / (np.linalg.norm(DOCKING_AXIS) + 1e-9)
                approach_dir = -n_app
                p_rel_final = p_chaser - p_target
                dist_long = -n_app @ p_rel_final
                rel_norm = np.linalg.norm(p_rel_final)
                cone_lhs = np.linalg.norm(p_rel_final) * np.cos(np.deg2rad(CONE_ANGLE))
                cone_violation_now = cone_lhs - dist_long
                cone_ok = (dist_long >= 0.0 and cone_violation_now <= CONE_EVAL_MARGIN)
                cos_theta_now = np.clip(dist_long / (rel_norm + 1e-9), -1.0, 1.0)
                theta_now_deg = float(np.rad2deg(np.arccos(cos_theta_now)))
                rel_vel = obs[0][10:13] - true_state[3:6]
                if docking_phase == 1 and (not cone_ok) and (sim_t - last_cone_warn_t >= 0.5):
                    print(
                        f"[ConeCheck] t={sim_t:.2f}s | violated | "
                        f"theta={theta_now_deg:.2f}deg, violation={cone_violation_now:.4f}, dist_long={dist_long:.4f}"
                    )
                    last_cone_warn_t = sim_t
                if alpha_obs < DCOL_OBS_HARD_ALPHA and (sim_t - last_dcol_warn_t >= 0.5):
                    print(
                        f"[DCOL] t={sim_t:.2f}s | alpha_obs={alpha_obs:.3f} < {DCOL_OBS_HARD_ALPHA:.3f} | "
                        f"activating obstacle guard"
                    )
                    last_dcol_warn_t = sim_t
                
                # Phase switch: once the chaser reaches a stable pre-dock point above target,
                # switch to terminal cone-constrained docking.
                if docking_phase == 0:
                    staging_pt = p_target + DOCK_OFFSET
                    stage_err = np.linalg.norm(p_chaser - staging_pt)
                    z_above = p_chaser[2] - p_target[2]
                    if stage_err < PRE_DOCK_SWITCH_DIST and z_above > PRE_DOCK_MIN_ABOVE:
                        docking_phase = 1
                        DOCK_OFFSET = np.zeros(3)
                        dock_standoff = min(PRE_DOCK_OFFSET_DIST, max(0.05, dist_long))
                        dock_hold_counter = 0
                        early_stop_counter = 0
                        terminal_lock = False
                        planner.reset_warm_start()
                        curr_plan = None
                        plan_step_idx = 0
                        last_plan = -PLAN_INTERVAL
                        print(f"[Phase] t={sim_t:.2f}s | Switching to TERMINAL cone docking")

                if docking_phase == 1 and cone_ok and dist_long <= SLOW_DOCK_START_DIST:
                    terminal_lock = True

                # --- DOCKING COMPLETION CHECK ---
                rel_dist = np.linalg.norm(p_rel_final)
                rel_speed = np.linalg.norm(rel_vel)
                strict_success = (
                    docking_phase == 1 and
                    cone_ok and
                    rel_dist <= DOCK_REL_POS_TOL and
                    rel_speed <= DOCK_REL_VEL_TOL and
                    alpha_obs >= EARLY_STOP_DCOL_ALPHA
                )
                if strict_success:
                    dock_hold_counter += 1
                else:
                    dock_hold_counter = 0

                early_success = (
                    EARLY_STOP_ENABLE and
                    docking_phase == 1 and
                    cone_ok and
                    rel_dist <= EARLY_STOP_REL_POS_TOL and
                    rel_speed <= EARLY_STOP_REL_VEL_TOL and
                    alpha_obs >= EARLY_STOP_DCOL_ALPHA
                )
                if early_success:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0

                if docking_phase == 1 and (dock_hold_counter >= DOCK_HOLD_STEPS or early_stop_counter >= EARLY_STOP_HOLD_STEPS):
                    frozen = True
                    freeze_pos_c = p_chaser
                    freeze_pos_t = p_target
                    success_label = "STRICT_DOCK" if dock_hold_counter >= DOCK_HOLD_STEPS else "EARLY_CAPTURE"
                    tol_used = DOCK_REL_POS_TOL if success_label == "STRICT_DOCK" else EARLY_STOP_REL_POS_TOL

                    print(f"\n[COMPLETE] {success_label} SUCCESS")
                    print(
                        f">>> Stop time: {sim_t:.2f} s | rel_dist={rel_dist:.3f} m | "
                        f"rel_speed={rel_speed:.3f} m/s | alpha_obs={alpha_obs:.3f} | cone={cone_ok} <<<"
                    )
                    
                    # Capture final frame before freezing
                    history['t'].append(sim_t)
                    history['p_c'].append(obs[0][0:3].copy())
                    history['p_t'].append(true_state[0:3].copy())
                    history['v_c'].append(obs[0][10:13].copy())
                    history['action'].append(action[0].copy())
                    
                    plot_performance(history, docking_threshold=tol_used)
                    break

                # 3. CONTROL LOGIC
                dcol_guard_active = False
                if state == STATE_TRACKING:
                    # Backoff Condition
                    if USE_REACTIVE_BACKOFF and alpha_obs < ALPHA_LIMIT and curr_plan is None:
                        print(f"[ALERT] Obstacle Collision Risk! Backing Off.")
                        state = STATE_BACKING_OFF
                        backoff_start = p_chaser.copy()
                        vec = p_chaser - P_OBS
                        vec = vec / (np.linalg.norm(vec)+1e-6)
                        backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                        backoff_t_start = time.time()
                        p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
                    else:
                        # Plan
                        res = planner.get()
                        if res is not None:
                            curr_plan = res
                            plan_step_idx = 0
                            if curr_plan.get("solver") is not None and curr_plan.get("solver") != last_solver_name:
                                last_solver_name = curr_plan.get("solver")
                                print(f"[Planner] Active solver: {last_solver_name}")
                            if curr_plan.get("diag") is not None:
                                d = curr_plan["diag"]
                                print(
                                    "[Planner] Plan diag | "
                                    f"term_pos={d['term_pos_err']:.3f}, term_vel={d['term_vel_err']:.3f}, "
                                    f"d_start={d['start_target_dist']:.3f}, d_end={d['end_target_dist']:.3f}, "
                                    f"min_obs_margin={d['min_obs_margin']:.3f}, max_u_inf={d['max_u_inf']:.3f}, "
                                    f"cone={d['cone_enforced']}, cone_violation={d['max_cone_violation']:.4f}, "
                                    f"max_dlong_inc={d['max_dlong_increase']:.4f}"
                                )
                            curr_traj = curr_plan["x"]
                            draw_planned_trajectory(curr_traj, PYB)

                        should_replan = (docking_phase == 0) or ENABLE_TERMINAL_REPLANNING
                        if should_replan and sim_t - last_plan > PLAN_INTERVAL:
                            preds = get_plan_predictions(sim_t)
                            chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                            planner.request(
                                chaser_st,
                                preds,
                                dock_offset=DOCK_OFFSET,
                                enforce_cone=(docking_phase == 1),
                            )
                            last_plan = sim_t

                        # --- THE FIX: SMOOTH BUT IMMEDIATE DIVE ---
                        # if docking_phase == 1:
                        #     # Bypass the slow MPC completely.
                        #     # Feed the raw target position AND the target's exact velocity to the PID.
                        #     # The velocity feed-forward ensures the drone smoothly glides down 
                        #     # while perfectly matching the target's forward speed.
                        #     action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_target, true_state[3:6])
                        # else:
                            # Phase 0: Follow the safe MPC trajectory to the moving offset point
                        if docking_phase == 1:
                                # Terminal relative controller: match target velocity and close along docking axis.
                                if cone_ok:
                                    dock_standoff = max(0.0, dock_standoff - DOCK_DESCENT_RATE * env.CTRL_TIMESTEP)
                                desired_standoff = min(SLOW_DOCK_START_DIST, dock_standoff)
                                desired_rel = approach_dir * desired_standoff
                                rel_err = desired_rel - p_rel_final
                                pt = p_target + desired_rel
                                vt = true_state[3:6] + TERM_REL_KP * rel_err - TERM_REL_KD * rel_vel
                                if not cone_ok:
                                    # Cone recovery mode: project current position to cone and recenter laterally
                                    # before continuing descent.
                                    p_proj = project_to_docking_cone(
                                        p_chaser,
                                        p_target,
                                        DOCKING_AXIS,
                                        CONE_ANGLE,
                                        long_margin=RUNTIME_CONE_LONG_MARGIN
                                    )
                                    p_proj = project_to_obstacle_clearance(
                                        p_proj,
                                        P_OBS,
                                        planner.obs_clearance + RUNTIME_OBS_MARGIN
                                    )
                                    pt = p_proj
                                    v_rel_rec = TERM_CONE_RECOVERY_KP * (p_proj - p_chaser) - TERM_REL_KD * rel_vel
                                    v_long_rec = np.dot(v_rel_rec, approach_dir)
                                    v_perp_rec = v_rel_rec - v_long_rec * approach_dir
                                    v_perp_n = np.linalg.norm(v_perp_rec)
                                    if v_perp_n > (1.5 * RUNTIME_MAX_REL_LATERAL_SPEED):
                                        v_perp_rec = v_perp_rec * ((1.5 * RUNTIME_MAX_REL_LATERAL_SPEED) / (v_perp_n + 1e-9))
                                    # Keep recovery mostly lateral; avoid slow upward drift.
                                    v_long_rec = min(
                                        TERM_CONE_RECOVERY_MAX_UP,
                                        max(0.0, TERM_CONE_RECOVERY_LONG_GAIN * cone_violation_now)
                                    )
                                    vt = true_state[3:6] + v_perp_rec + v_long_rec * approach_dir
                                else:
                                    pt = enforce_runtime_constraints(
                                        pt,
                                        p_target,
                                        P_OBS,
                                        planner.obs_clearance + RUNTIME_OBS_MARGIN,
                                        DOCKING_AXIS,
                                        CONE_ANGLE,
                                        enforce_cone=True
                                    )
                                    # Clamp relative velocity in terminal mode to avoid outward drift.
                                    v_rel_cmd = vt - true_state[3:6]
                                    v_long = np.dot(v_rel_cmd, approach_dir)
                                    v_perp = v_rel_cmd - v_long * approach_dir
                                    v_perp_n = np.linalg.norm(v_perp)
                                    if v_perp_n > RUNTIME_MAX_REL_LATERAL_SPEED:
                                        v_perp = v_perp * (RUNTIME_MAX_REL_LATERAL_SPEED / (v_perp_n + 1e-9))
                                    v_long = min(v_long, 0.0)  # never command moving away along docking axis
                                    vt = true_state[3:6] + v_perp + v_long * approach_dir
                                pt, vt, dcol_guard_active, alpha_obs = enforce_dcol_obstacle_guard(
                                    pt, vt, p_chaser, obs[0][10:13], P_OBS, SAFETY_R, R_OBS
                                )
                                pt[2] = np.clip(pt[2], Z_MIN, Z_MAX)
                                v_norm = np.linalg.norm(vt)
                                if v_norm > V_REF_MAX_TERMINAL:
                                    vt = vt * (V_REF_MAX_TERMINAL / (v_norm + 1e-6))
                                action[0], _, _ = ctrl[0].computeControlFromState(
                                    control_timestep=env.CTRL_TIMESTEP,
                                    state=obs[0],
                                    target_pos=pt,
                                    target_vel=vt
                                )
                        elif curr_plan is not None:
                                curr_traj = curr_plan["x"]
                                curr_u = curr_plan["u"]
                                traj_pos = curr_traj[0:3, :].T
                                nearest_idx = int(np.argmin(np.linalg.norm(traj_pos - p_chaser[None, :], axis=1)))
                                idx_seed = max(plan_step_idx, nearest_idx)
                                idx = min(idx_seed + TRAJ_LOOKAHEAD_IDX, curr_traj.shape[1]-1)
                                pt = curr_traj[0:3, idx].copy()
                                vt = curr_traj[3:6, idx].copy()
                                # Use SCP u* as feed-forward during tracking.
                                if curr_u is not None and curr_u.shape[1] > 0:
                                    u_idx = min(idx, curr_u.shape[1]-1)
                                    u_ff = np.clip(curr_u[:, u_idx], -U_MAX, U_MAX)
                                    vt = vt + U_FF_GAIN * u_ff * env.CTRL_TIMESTEP
                                pt = enforce_runtime_constraints(
                                    pt,
                                    p_target,
                                    P_OBS,
                                    planner.obs_clearance + RUNTIME_OBS_MARGIN,
                                    DOCKING_AXIS,
                                    CONE_ANGLE,
                                    enforce_cone=False
                                )
                                pt, vt, dcol_guard_active, alpha_obs = enforce_dcol_obstacle_guard(
                                    pt, vt, p_chaser, obs[0][10:13], P_OBS, SAFETY_R, R_OBS
                                )
                                pt[2] = np.clip(pt[2], Z_MIN, Z_MAX)
                                v_norm = np.linalg.norm(vt)
                                if v_norm > V_REF_MAX:
                                    vt = vt * (V_REF_MAX / (v_norm + 1e-6))
                                if USE_SCP_DIRECT_FORCE_DEBUG and not dcol_guard_active:
                                    action[0] = hover_rpm_cmd.copy()
                                    if curr_u is not None and curr_u.shape[1] > 0:
                                        u_idx = min(idx, curr_u.shape[1]-1)
                                        u_cmd = np.clip(curr_u[:, u_idx], -U_MAX, U_MAX)
                                        direct_u_lpf = (1.0 - DIRECT_FORCE_ALPHA) * direct_u_lpf + DIRECT_FORCE_ALPHA * u_cmd
                                        f_world = DIRECT_FORCE_SCALE * env.M * direct_u_lpf
                                        p.applyExternalForce(env.DRONE_IDS[0], -1, f_world, p_chaser, p.WORLD_FRAME, PYB)
                                        p.addUserDebugLine(
                                            p_chaser,
                                            p_chaser + 2.0 * f_world,
                                            [0.0, 1.0, 1.0],
                                            2,
                                            lifeTime=0.1,
                                            physicsClientId=PYB
                                        )
                                else:
                                    action[0], _, _ = ctrl[0].computeControlFromState(
                                        control_timestep=env.CTRL_TIMESTEP,
                                        state=obs[0],
                                        target_pos=pt,
                                        target_vel=vt
                                    )
                                plan_step_idx = min(idx_seed + 1, curr_traj.shape[1]-1)
                        else:
                                # No trajectory yet: hold position instead of chasing target unsafely.
                                fallback_pt = obs[0][0:3].copy()
                                fallback_vt = np.zeros(3)
                                fallback_pt = enforce_runtime_constraints(
                                    fallback_pt,
                                    p_target,
                                    P_OBS,
                                    planner.obs_clearance + RUNTIME_OBS_MARGIN,
                                    DOCKING_AXIS,
                                    CONE_ANGLE,
                                    enforce_cone=(docking_phase == 1)
                                )
                                fallback_pt, fallback_vt, dcol_guard_active, alpha_obs = enforce_dcol_obstacle_guard(
                                    fallback_pt, fallback_vt, p_chaser, obs[0][10:13], P_OBS, SAFETY_R, R_OBS
                                )
                                fallback_pt[2] = np.clip(fallback_pt[2], Z_MIN, Z_MAX)
                                fb_v_norm = np.linalg.norm(fallback_vt)
                                if fb_v_norm > V_REF_MAX:
                                    fallback_vt = fallback_vt * (V_REF_MAX / (fb_v_norm + 1e-6))
                                action[0], _, _ = ctrl[0].computeControlFromState(
                                    control_timestep=env.CTRL_TIMESTEP,
                                    state=obs[0],
                                    target_pos=fallback_pt,
                                    target_vel=fallback_vt
                                )

                elif state == STATE_BACKING_OFF:
                    elapsed = time.time() - backoff_t_start
                    progress = min(elapsed / 2.0, 1.0)
                    k = progress * progress * (3 - 2 * progress)
                    setpoint = (1-k)*backoff_start + k*backoff_end
                    setpoint, v_set, dcol_guard_active, alpha_obs = enforce_dcol_obstacle_guard(
                        setpoint, np.zeros(3), p_chaser, obs[0][10:13], P_OBS, SAFETY_R, R_OBS
                    )
                    setpoint[2] = np.clip(setpoint[2], Z_MIN, Z_MAX)
                    v_set_n = np.linalg.norm(v_set)
                    if v_set_n > V_REF_MAX:
                        v_set = v_set * (V_REF_MAX / (v_set_n + 1e-6))
                    action[0], _, _ = ctrl[0].computeControlFromState(
                        control_timestep=env.CTRL_TIMESTEP,
                        state=obs[0],
                        target_pos=setpoint,
                        target_vel=v_set
                    )
                    
                    if progress >= 1.0:
                        state = STATE_REPLANNING
                        
                elif state == STATE_REPLANNING:
                    hold_pt, hold_vt, dcol_guard_active, alpha_obs = enforce_dcol_obstacle_guard(
                        p_chaser, np.zeros(3), p_chaser, obs[0][10:13], P_OBS, SAFETY_R, R_OBS
                    )
                    hold_pt[2] = np.clip(hold_pt[2], Z_MIN, Z_MAX)
                    hold_v_n = np.linalg.norm(hold_vt)
                    if hold_v_n > V_REF_MAX:
                        hold_vt = hold_vt * (V_REF_MAX / (hold_v_n + 1e-6))
                    action[0], _, _ = ctrl[0].computeControlFromState(
                        control_timestep=env.CTRL_TIMESTEP,
                        state=obs[0],
                        target_pos=hold_pt,
                        target_vel=hold_vt
                    )
                    if sim_t - last_plan > 0.1:
                        preds = get_plan_predictions(sim_t)
                        chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                        planner.request(
                            chaser_st,
                            preds,
                            dock_offset=DOCK_OFFSET,
                            enforce_cone=(docking_phase == 1),
                        )
                        last_plan = sim_t
                    
                    res = planner.get()
                    if res is not None:
                        res_x = res["x"]
                        if np.linalg.norm(res_x[0:3,0] - p_chaser) < 0.3:
                            curr_plan = res
                            p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                            state = STATE_TRACKING
                
                action[1] = np.zeros(4)
                
                # Print terminal output less frequently
                if i % 20 == 0:
                    dist = np.linalg.norm(p_rel_final)
                    rel_speed_now = np.linalg.norm(rel_vel)
                    p_str = "OFFSET" if docking_phase == 0 else "TERMINAL"
                    print(
                        f"{sim_t:05.2f} | Phase: {p_str} | Dist: {dist:.2f}m | "
                        f"RelV: {rel_speed_now:.2f}m/s | Theta: {theta_now_deg:.2f}deg | "
                        f"Cone: {cone_ok} | ConeViol: {cone_violation_now:.4f} | "
                        f"AlphaObs: {alpha_obs:.3f} | DCOLguard: {dcol_guard_active} | "
                        f"Lock: {terminal_lock} | Hold: {dock_hold_counter}/{DOCK_HOLD_STEPS} | "
                        f"Early: {early_stop_counter}/{EARLY_STOP_HOLD_STEPS}"
                    )
                
                # 4. LOGGING DATA (Only if not frozen)
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
        
        # Plot if time ran out without successfully docking
        if not frozen:
            print("[PLOTS] Time expired. Generating Performance Plots...")
            plot_performance(history)


if __name__ == "__main__":
    run()
