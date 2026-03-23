import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

SCP_SOLVER_ORDER = ("ECOS", "CLARABEL", "SCS")
ACCEPTABLE_SOLVER_STATUSES = {"optimal", "optimal_inaccurate"}


def solve_with_fallback(prob: cp.Problem, warm_start: bool = True) -> str:
    """Solve SCP subproblem with robust conic solver fallback."""
    installed = set(cp.installed_solvers())
    attempted = []
    last_status = None

    for solver_name in SCP_SOLVER_ORDER:
        if solver_name not in installed:
            continue
        attempted.append(solver_name)

        solver_kwargs = {}
        if solver_name == "SCS":
            solver_kwargs["max_iters"] = 6000
            solver_kwargs["eps"] = 1e-4

        try:
            prob.solve(solver=solver_name, warm_start=warm_start, **solver_kwargs)
        except cp.error.SolverError:
            continue

        last_status = prob.status
        if prob.status in ACCEPTABLE_SOLVER_STATUSES:
            return solver_name

    if not attempted:
        raise cp.error.SolverError(
            "No supported conic solver found. Install at least one of: ECOS, CLARABEL, SCS."
        )
    raise cp.error.SolverError(
        f"All SCP solvers failed or returned non-optimal status. Tried {attempted}, last status={last_status}."
    )


# ================= SYSTEM =================
dt = 0.1
N = 25  # Planning horizon

A_MAX = 15.0
V_MAX = 5.0

P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)
N_APP = np.array([0, 0, -1])
r_dock = 0.0

MAX_TILT = np.radians(25)
U_MIN = 2.0
U_MAX = 15.0
GRAVITY = 9.81

# 9D State: [px, py, pz, vx, vy, vz, phi(roll), theta(pitch), a_T(thrust)]
x0 = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, GRAVITY])
p_target_true = np.array([0.5, 0.0, 1.0])

# PID gains for reference tracking
KP_POS = np.array([2.4, 2.4, 3.2])
KD_VEL = np.array([1.8, 1.8, 2.2])
KI_POS = np.array([0.05, 0.05, 0.10])
INT_CLIP = np.array([1.5, 1.5, 1.2])


# ================= FULL NON-LINEAR DRONE DYNAMICS =================
def f_dyn(x, u):
    """Continuous non-linear drone dynamics used for motion rollout."""
    px, py, pz, vx, vy, vz, phi, theta, a_T = x
    phi_cmd, theta_cmd, a_cmd = u

    tau_rp = 0.1
    tau_t = 0.05

    return np.array(
        [
            vx,
            vy,
            vz,
            a_T * np.sin(theta),
            -a_T * np.sin(phi) * np.cos(theta),
            a_T * np.cos(phi) * np.cos(theta) - GRAVITY,
            (phi_cmd - phi) / tau_rp,
            (theta_cmd - theta) / tau_rp,
            (a_cmd - a_T) / tau_t,
        ]
    )


class PositionPID:
    def __init__(self, kp, kd, ki, int_clip):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.int_clip = int_clip
        self.int_err = np.zeros(3)

    def reset(self):
        self.int_err = np.zeros(3)

    def compute(self, pos_err, vel_err, dt_ctrl):
        self.int_err += pos_err * dt_ctrl
        self.int_err = np.clip(self.int_err, -self.int_clip, self.int_clip)
        return self.kp * pos_err + self.kd * vel_err + self.ki * self.int_err


def accel_to_attitude_thrust(acc_des):
    """Map desired inertial acceleration to [phi_cmd, theta_cmd, a_cmd]."""
    a_world = np.asarray(acc_des, dtype=float) + np.array([0.0, 0.0, GRAVITY])
    a_norm = np.linalg.norm(a_world)
    a_norm = max(a_norm, 1e-6)

    theta_cmd = np.arcsin(np.clip(a_world[0] / a_norm, -0.95, 0.95))
    cth = np.cos(theta_cmd)
    denom = max(a_norm * max(cth, 1e-3), 1e-6)
    phi_cmd = -np.arcsin(np.clip(a_world[1] / denom, -0.95, 0.95))

    phi_cmd = np.clip(phi_cmd, -MAX_TILT, MAX_TILT)
    theta_cmd = np.clip(theta_cmd, -MAX_TILT, MAX_TILT)
    a_cmd = np.clip(a_norm, U_MIN, U_MAX)

    return np.array([phi_cmd, theta_cmd, a_cmd])


# ================= INITIAL GUESS (Relative Frame) =================
# Relative state [p_rel(3), v_rel(3)]
X_nom = np.zeros((N, 6))
x_rel0 = np.hstack([x0[0:3] - p_target_true, x0[3:6]])

for k in range(N):
    al = k / (N - 1)
    X_nom[k, 0:3] = (1 - al) * x_rel0[0:3]
    X_nom[k, 1] += 0.5 * np.sin(np.pi * al)


# ================= ONLINE SCP + PID LOOP =================
SIM_MAX_STEPS = 80
TOL = 1e-3
MAX_ITERS = 10

x_hist = [x0.copy()]
u_hist = []
x_star_hist = [X_nom.copy()]
phase_hist = []
cost_history = []
delta_history = []
trust_history = []

x_true = x0.copy()
phase = 0
pid = PositionPID(KP_POS, KD_VEL, KI_POS, INT_CLIP)

print("Starting Static EKF + x_star SCP + PID Motion (Drone Dynamics)...")

for sim_step in range(SIM_MAX_STEPS):
    # 1) Static target estimate with sensor noise (EKF-like measurement stage)
    sensor_noise = np.random.normal(0, 0.02, 3)
    p_target_est = p_target_true + sensor_noise

    # Stop condition on true state
    dist_to_goal = np.linalg.norm(x_true[0:3] - p_target_true)
    vel_mag = np.linalg.norm(x_true[3:6])
    if dist_to_goal < 0.15 and vel_mag < 0.2:
        print(f"Goal Reached at step {sim_step}!")
        break

    # Current relative state w.r.t. estimated target
    x_rel_true = np.hstack([x_true[0:3] - p_target_est, x_true[3:6]])

    # 2) Phase logic
    dist_xy = np.linalg.norm(x_rel_true[0:2])
    if phase == 0 and dist_xy < 0.3:
        phase = 1
        print(f"[{sim_step*dt:.1f}s] FSM TRIGGER: Phase 1 (Top-Down Cone) Activated!")

    # 3) Warm-start x_star only
    if sim_step > 0:
        X_nom[:-1, :] = X_nom[1:, :]
        X_nom[-1, :] = X_nom[-2, :]

    trust_radius = 2.0
    scp_converged = False
    last_prob_value = np.nan
    last_delta = np.nan
    last_solver = "none"

    # 4) Inner SCP loop (state only)
    for it in range(MAX_ITERS):
        X = cp.Variable((N, 6))
        slack_cone = cp.Variable(N - 1, nonneg=True)
        slack_tar = cp.Variable(N - 1, nonneg=True)

        cost = 0
        con = [X[0, :] == x_rel_true]

        offset = np.array([0.0, 0.0, 0.5]) if phase == 0 else np.zeros(3)

        # Soft terminal objective (x_star only)
        cost += cp.sum_squares(X[-1, 0:3] - offset) * 600.0
        cost += cp.sum_squares(X[-1, 3:6]) * 400.0

        for k in range(N - 1):
            # Kinematic dynamics without explicit control variable
            con += [X[k + 1, 0:3] == X[k, 0:3] + 0.5 * dt * (X[k, 3:6] + X[k + 1, 3:6])]

            acc_k = (X[k + 1, 3:6] - X[k, 3:6]) / dt

            # Costs
            p_rel = X[k, 0:3]
            cost += cp.sum_squares(p_rel - offset) * 2.0
            cost += cp.sum_squares(X[k, 3:6]) * 1.0
            cost += cp.sum_squares(acc_k) * 0.4
            cost += cp.sum_squares(X[k, :] - X_nom[k, :]) * 0.4

            # Trust region and physical limits
            con += [cp.norm(X[k, :] - X_nom[k, :], np.inf) <= trust_radius]
            con += [cp.norm(acc_k, np.inf) <= A_MAX]
            con += [cp.norm(X[k + 1, 3:6], 2) <= V_MAX]

            # Linearized obstacle (relative frame)
            p_rel_nom = X_nom[k, 0:3]
            p_obs_rel = P_OBS - p_target_est
            v_obs = p_rel_nom - p_obs_rel
            d_obs = np.linalg.norm(v_obs) + 1e-8
            n_obs = v_obs / d_obs
            con += [n_obs @ (X[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

            if phase == 1:
                dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
                n_tar = (p_rel_nom / dist_tar_nom) * r_dock
                con += [n_tar @ p_rel >= r_dock - slack_tar[k]]
                con += [cp.norm(p_rel) * np.cos(THETA) <= -N_APP @ p_rel + slack_cone[k]]
                con += [slack_tar[k] == 0]
            else:
                con += [slack_tar[k] == 0]
                con += [slack_cone[k] == 0]

        con += [cp.norm(X[-1, :] - X_nom[-1, :], np.inf) <= trust_radius]
        cost += cp.sum(slack_cone) * 100.0
        cost += cp.sum(slack_tar) * 100.0

        prob = cp.Problem(cp.Minimize(cost), con)
        try:
            last_solver = solve_with_fallback(prob, warm_start=True)
        except cp.error.SolverError:
            trust_radius *= 0.5
            continue

        if X.value is None:
            trust_radius *= 0.5
            continue

        last_prob_value = float(prob.value) if prob.value is not None else np.nan
        delta = np.linalg.norm(X.value - X_nom, np.inf)
        last_delta = float(delta)

        X_nom = X.value.copy()

        if delta < TOL:
            scp_converged = True
            break

        trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0)

    # 5) PID motion from x_star only (no u_nom/u_star replay)
    ref_idx = min(1, N - 1)
    nxt_idx = min(2, N - 1)
    x_ref_rel = X_nom[ref_idx]
    x_ref_rel_next = X_nom[nxt_idx]

    pos_ref_abs = p_target_est + x_ref_rel[0:3]
    vel_ref_abs = x_ref_rel[3:6]
    acc_ff = (x_ref_rel_next[3:6] - x_ref_rel[3:6]) / max(dt, 1e-6)

    pos_err = pos_ref_abs - x_true[0:3]
    vel_err = vel_ref_abs - x_true[3:6]
    acc_fb = pid.compute(pos_err, vel_err, dt)

    acc_des = np.clip(acc_ff + acc_fb, -A_MAX, A_MAX)
    u_cmd = accel_to_attitude_thrust(acc_des)

    # Integrate true non-linear drone dynamics
    dt_sim = 0.01
    for _ in range(int(dt / dt_sim)):
        x_true = x_true + dt_sim * f_dyn(x_true, u_cmd)

    # Logging
    x_hist.append(x_true.copy())
    u_hist.append(u_cmd.copy())
    x_star_hist.append(X_nom.copy())
    phase_hist.append(phase)
    cost_history.append(last_prob_value)
    delta_history.append(last_delta)
    trust_history.append(trust_radius)

    cost_str = f"{last_prob_value:.1f}" if np.isfinite(last_prob_value) else "nan"
    print(
        f"Step {sim_step:02d} | Phase: {phase} | Cost: {cost_str} | "
        f"Dist: {dist_to_goal:.2f} | Solver: {last_solver} | SCP Converged: {scp_converged}"
    )


# ================= PLOTTING =================
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
phase_hist = np.array(phase_hist)
x_star_hist = np.array(x_star_hist)
time_steps = np.arange(x_hist.shape[0]) * dt
ctrl_steps = np.arange(u_hist.shape[0]) * dt

plt.style.use("seaborn-v0_8-darkgrid")
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(121, projection="3d")

traj = x_hist[:, 0:3]
ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b.-", linewidth=2, label="Executed Chaser Traj")
ax1.plot(x0[0], x0[1], x0[2], "go", markersize=8, label="Start")
ax1.plot(p_target_true[0], p_target_true[1], p_target_true[2], "r*", markersize=12, label="True Target")

u_sph, v_sph = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
x_sph = P_OBS[0] + R_OBS * np.cos(u_sph) * np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS * np.sin(u_sph) * np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS * np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color="r", alpha=0.3)

ax1.set_title("Static EKF: x_star SCP + PID Motion")
ax1.legend()

ax2 = fig1.add_subplot(122)
dist_arr = np.linalg.norm(x_hist[:-1, 0:3] - p_target_true, axis=1)
ax2.plot(ctrl_steps, dist_arr, "m-", linewidth=2, label="Distance to Target")
ax2.axhline(0.3, color="k", linestyle="--", label="FSM Trigger Radius")
ax2.fill_between(
    ctrl_steps,
    0,
    max(dist_arr) if len(dist_arr) else 1.0,
    where=(phase_hist == 1),
    color="cyan",
    alpha=0.2,
    transform=ax2.get_xaxis_transform(),
    label="Phase 1 Active",
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Distance (m)")
ax2.set_title("FSM Tracking")
ax2.legend()

fig2, (ax_u1, ax_u2, ax_u3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
if u_hist.shape[0] > 0:
    ax_u1.plot(ctrl_steps, np.degrees(u_hist[:, 0]), "b")
    ax_u2.plot(ctrl_steps, np.degrees(u_hist[:, 1]), "g")
    ax_u3.plot(ctrl_steps, u_hist[:, 2], "r")
ax_u1.set_ylabel("phi_cmd (deg)")
ax_u2.set_ylabel("theta_cmd (deg)")
ax_u3.set_ylabel("a_cmd (m/s^2)")
ax_u3.set_xlabel("Time (s)")
ax_u1.set_title("Applied PID Commands")

plt.tight_layout()
plt.show()
