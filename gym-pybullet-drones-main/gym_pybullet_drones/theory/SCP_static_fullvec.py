import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ================= SYSTEM =================
dt = 0.1
N = 25
time_steps = np.arange(N) * dt
ctrl_steps = np.arange(N - 1) * dt

# Crazyflie 2X parameters (from assets/cf2x.urdf)
M = 0.027
G = 9.8
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
J_INV = np.linalg.inv(J)
L_ARM = 0.0397
KF = 3.16e-10
KM = 7.94e-12
THRUST2WEIGHT_RATIO = 2.25

F_HOVER = M * G
F_MAX = THRUST2WEIGHT_RATIO * F_HOVER
TAU_XY_MAX = (L_ARM / np.sqrt(2.0)) * F_MAX
TAU_Z_MAX = 0.5 * (KM / KF) * F_MAX

V_MAX = 5.0
OMEGA_MAX = 6.0

P_OBS = np.array([-1.0, 0.0, 1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30.0)
N_APP = np.array([0.0, 0.0, -1.0])

r_c = 0.1
r_t = 0.1
alpha_min = 1.05

MAX_TILT_DEG = 35.0
MAX_TILT_RAD = np.radians(MAX_TILT_DEG)
# For q = [qw, qx, qy, qz], tilt angle to world z satisfies:
# cos(tilt) = 1 - 2*(qx^2 + qy^2)
MAX_TILT_QXY_SQ = 0.5 * (1.0 - np.cos(MAX_TILT_RAD))

# Objective weights
W_POS = 10.0
W_VEL = 1.0
W_THRUST = 4.0
W_TAU = 60.0
W_Q_TRACK = 0.5
W_OMEGA_TRACK = 0.3
W_DYN_SLACK = 2.0e4
W_QNORM_SLACK = 2.0e4

x0_pv = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0])
q0 = np.array([1.0, 0.0, 0.0, 0.0])
omega0 = np.zeros(3)
x0 = np.hstack([x0_pv, q0, omega0])

p_target = np.array([0.5, 0.0, 1.0])
q_target = np.array([1.0, 0.0, 0.0, 0.0])


# ================= HELPERS =================
def alpha(pc: np.ndarray) -> float:
    return np.linalg.norm(pc - p_target) / (r_c + r_t)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(q)
    if nrm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / nrm


def quat_omega_matrix(omega: np.ndarray) -> np.ndarray:
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ]
    )


def quat_rate_jacobian_w(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ]
    )


def quat_tilt_deg_from_wxyz(q: np.ndarray) -> float:
    qn = normalize_quat(q)
    qx = qn[1]
    qy = qn[2]
    cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
    cos_tilt = float(np.clip(cos_tilt, -1.0, 1.0))
    return np.degrees(np.arccos(cos_tilt))


def body_z_world_from_quat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ]
    )


def body_z_jacobian_quat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [2.0 * qy, 2.0 * qz, 2.0 * qw, 2.0 * qx],
            [-2.0 * qx, -2.0 * qw, 2.0 * qz, 2.0 * qy],
            [0.0, -4.0 * qx, -4.0 * qy, 0.0],
        ]
    )


def omega_coriolis(omega: np.ndarray) -> np.ndarray:
    return np.cross(omega, J @ omega)


def omega_coriolis_jacobian(omega: np.ndarray) -> np.ndarray:
    p, q, r = omega
    ixx, iyy, izz = J[0, 0], J[1, 1], J[2, 2]
    return np.array(
        [
            [0.0, (izz - iyy) * r, (izz - iyy) * q],
            [(ixx - izz) * r, 0.0, (ixx - izz) * p],
            [(iyy - ixx) * q, (iyy - ixx) * p, 0.0],
        ]
    )


def quat_from_rotmat(rot: np.ndarray) -> np.ndarray:
    tr = np.trace(rot)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return normalize_quat(np.array([qw, qx, qy, qz]))


def quat_from_body_z_and_yaw(b3_world: np.ndarray, yaw_des_rad: float = 0.0) -> np.ndarray:
    z_body = b3_world / (np.linalg.norm(b3_world) + 1e-9)
    x_c = np.array([np.cos(yaw_des_rad), np.sin(yaw_des_rad), 0.0])
    y_body = np.cross(z_body, x_c)
    if np.linalg.norm(y_body) < 1e-7:
        x_c = np.array([1.0, 0.0, 0.0])
        y_body = np.cross(z_body, x_c)
    y_body /= np.linalg.norm(y_body) + 1e-9
    x_body = np.cross(y_body, z_body)
    x_body /= np.linalg.norm(x_body) + 1e-9
    rot = np.column_stack((x_body, y_body, z_body))
    return quat_from_rotmat(rot)


# ================= INITIAL GUESS =================
x_nom = np.zeros((N, 13))
for k in range(N):
    al = k / (N - 1)
    pos = (1.0 - al) * x0_pv[0:3] + al * p_target
    pos[1] += 0.8 * np.sin(np.pi * al)
    pos[2] += 0.5 * np.sin(np.pi * al)
    x_nom[k, 0:3] = pos

for k in range(N - 1):
    x_nom[k, 3:6] = (x_nom[k + 1, 0:3] - x_nom[k, 0:3]) / dt
x_nom[-1, 3:6] = 0.0

for k in range(N):
    if k < N - 1:
        a_guess = (x_nom[k + 1, 3:6] - x_nom[k, 3:6]) / dt
    else:
        a_guess = np.zeros(3)
    b3_guess = a_guess + np.array([0.0, 0.0, G])
    if np.linalg.norm(b3_guess) < 1e-8:
        b3_guess = np.array([0.0, 0.0, 1.0])
    x_nom[k, 6:10] = quat_from_body_z_and_yaw(b3_guess, yaw_des_rad=0.0)

x_nom[:, 10:13] = 0.0
x_nom[0, 6:10] = q0
x_nom[-1, 6:10] = q_target

# Control: [total_thrust_N, tau_x, tau_y, tau_z]
u_nom = np.zeros((N - 1, 4))
for k in range(N - 1):
    a_guess = (x_nom[k + 1, 3:6] - x_nom[k, 3:6]) / dt
    b3 = body_z_world_from_quat(x_nom[k, 6:10])
    f_guess = M * np.dot(a_guess + np.array([0.0, 0.0, G]), b3)
    u_nom[k, 0] = np.clip(f_guess, 0.0, F_MAX)

# ================= TRACKING ARRAYS =================
cost_history = []
delta_history = []
trust_history = []

# ================= SCP LOOP =================
trust_radius = 2.0
TOL = 1e-3
MAX_ITERS = 20

print("Starting 13D SCP with coupled translational-attitude dynamics...")
for it in range(MAX_ITERS):
    x = cp.Variable((N, 13))
    u = cp.Variable((N - 1, 4))

    slack_cone = cp.Variable(N - 1, nonneg=True)
    slack_tar = cp.Variable(N - 1, nonneg=True)
    slack_qnorm = cp.Variable(N, nonneg=True)

    # Virtual-control slack variables for linearized dynamics.
    nu_p = cp.Variable((N - 1, 3))
    nu_v = cp.Variable((N - 1, 3))
    nu_q = cp.Variable((N - 1, 4))
    nu_w = cp.Variable((N - 1, 3))

    cost = 0
    con = [x[0, :] == x0]

    con += [x[-1, 0:3] == p_target]
    con += [x[-1, 3:6] == np.zeros(3)]
    con += [x[-1, 6:10] == q_target]
    con += [x[-1, 10:13] == np.zeros(3)]

    for k in range(N - 1):
        q_nom_k = normalize_quat(x_nom[k, 6:10])
        w_nom_k = x_nom[k, 10:13]
        f_nom_k = float(u_nom[k, 0])
        tau_nom_k = u_nom[k, 1:4]

        # Coupled translational dynamics: a = (f/m)*b3(q) - g*e3 (linearized)
        b3_nom = body_z_world_from_quat(q_nom_k)
        db3_dq = body_z_jacobian_quat(q_nom_k)

        a_nom = (f_nom_k / M) * b3_nom - np.array([0.0, 0.0, G])
        j_q_acc = (f_nom_k / M) * db3_dq
        j_f_acc = b3_nom / M
        c_acc = a_nom - j_q_acc @ q_nom_k - j_f_acc * f_nom_k

        a_lin = j_q_acc @ x[k, 6:10] + j_f_acc * u[k, 0] + c_acc

        con += [
            x[k + 1, 0:3]
            == x[k, 0:3] + dt * x[k, 3:6] + 0.5 * dt**2 * a_lin + nu_p[k, :]
        ]
        con += [x[k + 1, 3:6] == x[k, 3:6] + dt * a_lin + nu_v[k, :]]

        # Angular-rate rigid body dynamics: w_dot = J^-1 (tau - w x Jw)
        cori_nom = omega_coriolis(w_nom_k)
        d_cori_d_w = omega_coriolis_jacobian(w_nom_k)

        j_w_dyn = -J_INV @ d_cori_d_w
        j_tau_dyn = J_INV
        wdot_nom = J_INV @ (tau_nom_k - cori_nom)
        c_w = wdot_nom - j_w_dyn @ w_nom_k - j_tau_dyn @ tau_nom_k

        con += [
            x[k + 1, 10:13]
            == x[k, 10:13] + dt * (j_w_dyn @ x[k, 10:13] + j_tau_dyn @ u[k, 1:4] + c_w) + nu_w[k, :]
        ]

        # Quaternion kinematics linearized around nominal
        j_q = 0.5 * quat_omega_matrix(w_nom_k)
        j_w = 0.5 * quat_rate_jacobian_w(q_nom_k)
        f_q_nom = 0.5 * quat_omega_matrix(w_nom_k) @ q_nom_k
        c_q = f_q_nom - j_q @ q_nom_k - j_w @ w_nom_k

        con += [
            x[k + 1, 6:10]
            == x[k, 6:10] + dt * (j_q @ x[k, 6:10] + j_w @ x[k, 10:13] + c_q) + nu_q[k, :]
        ]

        # Soft linearized unit quaternion constraint.
        qnorm_err = 2.0 * q_nom_k @ x[k, 6:10] - q_nom_k @ q_nom_k - 1.0
        con += [cp.abs(qnorm_err) <= slack_qnorm[k]]

        # Objective
        cost += W_POS * cp.sum_squares(x[k, 0:3] - p_target)
        cost += W_VEL * cp.sum_squares(x[k, 3:6])
        cost += W_THRUST * cp.sum_squares(u[k, 0] - F_HOVER)
        cost += W_TAU * cp.sum_squares(u[k, 1:4])
        cost += W_Q_TRACK * cp.sum_squares(x[k, 6:10] - q_target)
        cost += W_OMEGA_TRACK * cp.sum_squares(x[k, 10:13])

        # Trust region
        con += [cp.norm(x[k, :] - x_nom[k, :], np.inf) <= trust_radius]

        p_nom = x_nom[k, 0:3]

        # Obstacle avoidance
        v_obs = p_nom - P_OBS
        d_obs = np.linalg.norm(v_obs) + 1e-8
        n_obs = v_obs / d_obs
        con += [n_obs @ (x[k, 0:3] - P_OBS) >= R_OBS + R_SAFE]

        p_rel_nom = x_nom[k, 0:3]

        p_obs_rel = P_OBS
        v_obs = p_rel_nom - p_obs_rel
        d_obs = np.linalg.norm(v_obs) + 1e-8
        n_obs = v_obs / d_obs
        con += [n_obs @ (x[k, 0:3] - p_obs_rel) >= R_OBS + R_SAFE]

        dist_xy = np.linalg.norm(p_rel_nom[0:2])
        if dist_xy < 1.5:
            dist_tar_nom = np.linalg.norm(p_rel_nom) + 1e-8
            n_tar = (p_rel_nom / dist_tar_nom) * (r_c + r_t)
            con += [n_tar @ x[k, 0:3] >= (r_c + r_t) * alpha_min - slack_tar[k]]
            con += [cp.norm(x[k, 0:3]) * np.cos(THETA) <= -N_APP @ x[k, 0:3] + slack_cone[k]]
        else:
            con += [slack_tar[k] == 0.0]
            con += [slack_cone[k] == 0.0]

        # Physical limits
        con += [u[k, 0] >= 0.0]
        con += [u[k, 0] <= F_MAX]
        con += [cp.abs(u[k, 1]) <= TAU_XY_MAX]
        con += [cp.abs(u[k, 2]) <= TAU_XY_MAX]
        con += [cp.abs(u[k, 3]) <= TAU_Z_MAX]

        con += [cp.norm(x[k + 1, 3:6], 2) <= V_MAX]
        con += [cp.norm(x[k + 1, 10:13], 2) <= OMEGA_MAX]

        con += [cp.norm(x[k + 1, 6:10], np.inf) <= 1.2]
        con += [cp.sum_squares(x[k + 1, 7:9]) <= MAX_TILT_QXY_SQ]

    con += [cp.norm(x[-1, :] - x_nom[-1, :], np.inf) <= trust_radius]
    q_nom_end = normalize_quat(x_nom[-1, 6:10])
    qnorm_err_end = 2.0 * q_nom_end @ x[-1, 6:10] - q_nom_end @ q_nom_end - 1.0
    con += [cp.abs(qnorm_err_end) <= slack_qnorm[-1]]
    con += [cp.sum_squares(x[-1, 7:9]) <= MAX_TILT_QXY_SQ]

    cost += 200.0 * cp.sum(slack_cone)
    cost += 200.0 * cp.sum(slack_tar)
    cost += W_QNORM_SLACK * cp.sum(slack_qnorm)
    cost += W_DYN_SLACK * (
        cp.sum_squares(nu_p) + cp.sum_squares(nu_v) + cp.sum_squares(nu_q) + cp.sum_squares(nu_w)
    )

    prob = cp.Problem(cp.Minimize(cost), con)

    solved = False
    for solver in (cp.ECOS, cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver)
            solved = True
            break
        except Exception:
            continue

    if (not solved) or (prob.status not in ["optimal", "optimal_inaccurate"]):
        print(f"Iter {it}: Infeasible. Shrinking trust region.")
        trust_radius *= 0.5
        continue

    x_next = x.value.copy()
    u_next = u.value.copy()

    for k in range(N):
        x_next[k, 6:10] = normalize_quat(x_next[k, 6:10])

    delta = np.linalg.norm(x_next - x_nom, np.inf)

    nu_max = max(
        np.max(np.abs(nu_p.value)),
        np.max(np.abs(nu_v.value)),
        np.max(np.abs(nu_q.value)),
        np.max(np.abs(nu_w.value)),
    )
    qnorm_slack_max = float(np.max(slack_qnorm.value))

    cost_history.append(float(prob.value))
    delta_history.append(float(delta))
    trust_history.append(float(trust_radius))

    print(
        f"Iter {it} | Cost: {prob.value:.1f} | delta: {delta:.4f} | trust: {trust_radius:.4f} "
        f"| dyn_slack_inf: {nu_max:.3e} | qnorm_slack: {qnorm_slack_max:.3e}"
    )

    x_nom = x_next
    u_nom = u_next

    if delta < TOL:
        print(">>> 13D SCP CONVERGED SUCCESSFULLY <<<")
        break

    trust_radius = float(np.clip(1.1 * trust_radius, 0.1, 3.0))


# ====================================================================
# ========================= PLOTTING DASHBOARDS ======================
# ====================================================================
plt.style.use("seaborn-v0_8-darkgrid")
traj = x_nom[:, 0:3]

# Figure 1: 3D trajectory
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b.-", linewidth=3, label="Optimized Chaser Traj")
ax1.plot(x0[0], x0[1], x0[2], "go", markersize=8, label="Start")
ax1.plot(p_target[0], p_target[1], p_target[2], "r*", markersize=12, label="Target")

u_sph, v_sph = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
x_sph = P_OBS[0] + R_OBS * np.cos(u_sph) * np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS * np.sin(u_sph) * np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS * np.cos(v_sph)
ax1.plot_surface(x_sph, y_sph, z_sph, color="r", alpha=0.3)

x_tar = p_target[0] + (r_c + r_t) * np.cos(u_sph) * np.sin(v_sph)
y_tar = p_target[1] + (r_c + r_t) * np.sin(u_sph) * np.sin(v_sph)
z_tar = p_target[2] + (r_c + r_t) * np.cos(v_sph)
ax1.plot_surface(x_tar, y_tar, z_tar, color="g", alpha=0.2)
ax1.set_title("DCOL Coupled Dynamics Trajectory")
ax1.legend()

# Figure 2: Thrust and speed
fig2, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax_u.plot(ctrl_steps, u_nom[:, 0], "tab:blue", label="thrust")
ax_u.axhline(F_HOVER, color="k", linestyle="--", label="hover thrust")
ax_u.axhline(F_MAX, color="r", linestyle="--", label="max thrust")
ax_u.set_ylabel("Thrust (N)")
ax_u.set_title("Coupled Translational Command")
ax_u.legend(loc="upper right")

v_norms = np.linalg.norm(x_nom[:, 3:6], axis=1)
ax_v.plot(time_steps, v_norms, "purple", linewidth=2, label="speed")
ax_v.axhline(V_MAX, color="k", linestyle="--", label="speed max")
ax_v.set_xlabel("Time (s)")
ax_v.set_ylabel("Speed (m/s)")
ax_v.legend(loc="upper right")

# Figure 3: Torque, omega, tilt
fig3, (ax_tau, ax_w, ax_tilt) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax_tau.plot(ctrl_steps, u_nom[:, 1], "tab:orange", label="tau_x")
ax_tau.plot(ctrl_steps, u_nom[:, 2], "tab:green", label="tau_y")
ax_tau.plot(ctrl_steps, u_nom[:, 3], "tab:blue", label="tau_z")
ax_tau.axhline(TAU_XY_MAX, color="k", linestyle="--")
ax_tau.axhline(-TAU_XY_MAX, color="k", linestyle="--")
ax_tau.set_ylabel("Torque (N m)")
ax_tau.set_title("Angular Commands")
ax_tau.legend(loc="upper right")

omega_norms = np.linalg.norm(x_nom[:, 10:13], axis=1)
ax_w.plot(time_steps, omega_norms, "tab:red", linewidth=2, label="omega")
ax_w.axhline(OMEGA_MAX, color="k", linestyle="--", label="omega max")
ax_w.set_ylabel("Angular rate (rad/s)")
ax_w.legend(loc="upper right")

tilt_hist_deg = np.array([quat_tilt_deg_from_wxyz(q) for q in x_nom[:, 6:10]])
ax_tilt.plot(time_steps, tilt_hist_deg, "tab:purple", linewidth=2, label="tilt")
ax_tilt.axhline(MAX_TILT_DEG, color="k", linestyle="--", label="tilt max")
ax_tilt.set_ylabel("Tilt (deg)")
ax_tilt.set_xlabel("Time (s)")
ax_tilt.set_title("Tilt Constraint")
ax_tilt.legend(loc="upper right")

# Figure 4: Convergence
fig4, (ax_c, ax_d, ax_t) = plt.subplots(1, 3, figsize=(12, 4))
iters = range(1, len(cost_history) + 1)

ax_c.plot(iters, cost_history, "mo-", linewidth=2)
ax_c.set_title("Objective cost")
ax_c.set_xlabel("Iteration")
ax_c.set_ylabel("Cost")

ax_d.semilogy(iters, delta_history, "co-", linewidth=2)
ax_d.axhline(TOL, color="k", linestyle="--", label="Tolerance")
ax_d.set_title("Trajectory change")
ax_d.set_xlabel("Iteration")
ax_d.set_ylabel("Delta")
ax_d.legend()

ax_t.plot(iters, trust_history, "yo-", linewidth=2)
ax_t.set_title("Trust region")
ax_t.set_xlabel("Iteration")
ax_t.set_ylabel("Radius")

plt.tight_layout()
plt.show()
