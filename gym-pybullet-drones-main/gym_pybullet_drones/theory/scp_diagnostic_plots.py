import numpy as np
import matplotlib.pyplot as plt


def _as_array(x, dtype=float):
    if x is None:
        return None
    arr = np.asarray(x, dtype=dtype)
    return arr


def _expand_target(target_arr, n_state):
    if target_arr is None:
        return np.zeros((n_state, 3))
    if target_arr.ndim == 1:
        return np.tile(target_arr.reshape(1, 3), (n_state, 1))
    if target_arr.shape[0] < n_state:
        pad = np.tile(target_arr[-1], (n_state - target_arr.shape[0], 1))
        return np.vstack([target_arr, pad])
    return target_arr[:n_state]


def _expand_ctrl(arr, n_ctrl):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < n_ctrl:
        pad = np.tile(arr[-1], (n_ctrl - arr.shape[0], 1))
        return np.vstack([arr, pad])
    return arr[:n_ctrl]


def _cone_angle_deg(p_rel, n_app):
    dist = np.linalg.norm(p_rel, axis=1)
    angles = np.zeros_like(dist)
    valid = dist > 1e-8
    if np.any(valid):
        cos_phi = np.sum((-n_app.reshape(1, 3)) * p_rel[valid], axis=1) / dist[valid]
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(cos_phi))
    return angles


def plot_scp_diagnostics(
    title_prefix,
    dt,
    x_hist,
    u_hist,
    phase_hist,
    target_pos_hist,
    obstacle_pos,
    r_obs,
    r_safe,
    theta,
    n_app,
    v_max,
    pos_ref_hist=None,
    vel_ref_hist=None,
    pos_err_hist=None,
    vel_err_hist=None,
    trust_history=None,
    cost_history=None,
    delta_history=None,
    virtual_norm_history=None,
    iter_cost_history=None,
    iter_trust_history=None,
    iter_virtual_norm_history=None,
    obstacle_residual_history=None,
    cone_residual_history=None,
    target_vel_hist=None,
):
    x_hist = _as_array(x_hist)
    u_hist = _as_array(u_hist)
    phase_hist = _as_array(phase_hist, dtype=int)

    if x_hist is None or x_hist.ndim != 2 or x_hist.shape[0] < 2:
        raise ValueError("x_hist must be a 2D array with at least two states.")

    n_state = x_hist.shape[0]
    n_ctrl = n_state - 1
    t_state = np.arange(n_state) * dt
    t_ctrl = np.arange(n_ctrl) * dt

    if u_hist is None:
        u_hist = np.zeros((n_ctrl, 3))
    if u_hist.ndim == 1:
        u_hist = u_hist.reshape(-1, 1)
    if u_hist.shape[0] < n_ctrl:
        pad = np.tile(u_hist[-1], (n_ctrl - u_hist.shape[0], 1))
        u_hist = np.vstack([u_hist, pad])
    else:
        u_hist = u_hist[:n_ctrl]

    if phase_hist is None:
        phase_hist = np.zeros(n_ctrl, dtype=int)
    else:
        phase_hist = phase_hist.flatten()
        if phase_hist.size < n_ctrl:
            phase_hist = np.concatenate([phase_hist, np.full(n_ctrl - phase_hist.size, phase_hist[-1])])
        phase_hist = phase_hist[:n_ctrl]

    target_pos_hist = _expand_target(_as_array(target_pos_hist), n_state)

    if target_vel_hist is None:
        target_vel_hist = np.zeros((n_state, 3))
    else:
        target_vel_hist = _expand_target(_as_array(target_vel_hist), n_state)

    pos_ref_hist = _expand_ctrl(pos_ref_hist, n_ctrl)
    vel_ref_hist = _expand_ctrl(vel_ref_hist, n_ctrl)
    pos_err_hist = _expand_ctrl(pos_err_hist, n_ctrl)
    vel_err_hist = _expand_ctrl(vel_err_hist, n_ctrl)

    if pos_err_hist is None:
        if pos_ref_hist is not None:
            pos_err_hist = pos_ref_hist - x_hist[:-1, 0:3]
        else:
            pos_err_hist = x_hist[:-1, 0:3] - target_pos_hist[:-1]

    if vel_err_hist is None:
        if vel_ref_hist is not None:
            vel_err_hist = vel_ref_hist - x_hist[:-1, 3:6]
        else:
            vel_err_hist = x_hist[:-1, 3:6] - target_vel_hist[:-1]

    obstacle_residual_history = _as_array(obstacle_residual_history)
    cone_residual_history = _as_array(cone_residual_history)
    trust_history = _as_array(trust_history)
    cost_history = _as_array(cost_history)
    delta_history = _as_array(delta_history)
    virtual_norm_history = _as_array(virtual_norm_history)
    iter_cost_history = _as_array(iter_cost_history)
    iter_trust_history = _as_array(iter_trust_history)
    iter_virtual_norm_history = _as_array(iter_virtual_norm_history)

    traj = x_hist[:, 0:3]
    dist_target = np.linalg.norm(traj - target_pos_hist, axis=1)
    p_rel = traj - target_pos_hist
    cone_angle_deg = _cone_angle_deg(p_rel, n_app)
    obs_clearance = np.linalg.norm(traj - obstacle_pos.reshape(1, 3), axis=1) - (r_obs + r_safe)
    cone_residual = (-p_rel @ n_app.reshape(3, 1)).flatten() - np.linalg.norm(p_rel, axis=1) * np.cos(theta)

    if obstacle_residual_history is None:
        obstacle_residual_history = obs_clearance[:-1]
    if cone_residual_history is None:
        cone_residual_history = cone_residual[:-1]

    if obstacle_residual_history.ndim == 0:
        obstacle_residual_history = np.full(n_ctrl, float(obstacle_residual_history))
    if cone_residual_history.ndim == 0:
        cone_residual_history = np.full(n_ctrl, float(cone_residual_history))
    obstacle_residual_history = obstacle_residual_history[:n_ctrl]
    cone_residual_history = cone_residual_history[:n_ctrl]

    plt.style.use("seaborn-v0_8-darkgrid")

    # 1) 3D Trajectory + Path tracking + obstacle visualization
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(121, projection="3d")
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b.-", linewidth=2, label="Executed path")
    if pos_ref_hist is not None:
        ax1.plot(
            pos_ref_hist[:, 0],
            pos_ref_hist[:, 1],
            pos_ref_hist[:, 2],
            "k--",
            linewidth=1.5,
            alpha=0.9,
            label="Reference path",
        )
    ax1.plot(target_pos_hist[:, 0], target_pos_hist[:, 1], target_pos_hist[:, 2], "r-", linewidth=2, label="Target")
    ax1.plot(traj[0, 0], traj[0, 1], traj[0, 2], "go", markersize=8, label="Start")
    ax1.plot(target_pos_hist[-1, 0], target_pos_hist[-1, 1], target_pos_hist[-1, 2], "r*", markersize=11, label="Dock")

    u_sph, v_sph = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    x_sph = obstacle_pos[0] + r_obs * np.cos(u_sph) * np.sin(v_sph)
    y_sph = obstacle_pos[1] + r_obs * np.sin(u_sph) * np.sin(v_sph)
    z_sph = obstacle_pos[2] + r_obs * np.cos(v_sph)
    ax1.plot_surface(x_sph, y_sph, z_sph, color="tomato", alpha=0.25, linewidth=0)
    ax1.set_title(f"{title_prefix}: 3D Trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_zlabel("z (m)")
    ax1.legend(loc="best")

    # 2) FSM plot (distance + phase)
    ax2 = fig1.add_subplot(122)
    ax2.plot(t_state, dist_target, "m-", linewidth=2, label="Distance to target")
    ax2.axhline(0.3, color="k", linestyle="--", linewidth=1.0, label="Phase trigger radius")
    ax2.fill_between(
        t_ctrl,
        0,
        1,
        where=(phase_hist == 1),
        transform=ax2.get_xaxis_transform(),
        color="cyan",
        alpha=0.15,
        label="Phase 1 active",
    )
    ax2.set_title("FSM Distance and Phase Activation")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.legend(loc="best")
    fig1.tight_layout()

    # 3) Safety plots
    fig2, (ax21, ax22) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax21.plot(t_state, obs_clearance, "r-", linewidth=2, label="Obstacle clearance")
    ax21.axhline(0.0, color="k", linestyle="--", linewidth=1.0, label="Safety boundary")
    ax21.set_ylabel("Clearance margin (m)")
    ax21.set_title("Safety: Obstacle Clearance")
    ax21.legend(loc="best")

    ax22.plot(t_state, cone_angle_deg, "c-", linewidth=2, label="Cone angle")
    ax22.axhline(np.degrees(theta), color="k", linestyle="--", linewidth=1.0, label="Cone limit")
    ax22.set_xlabel("Time (s)")
    ax22.set_ylabel("Angle (deg)")
    ax22.set_title("Safety: Cone Angle")
    ax22.legend(loc="best")
    fig2.tight_layout()

    # 4) State tracking error
    fig3, ax3 = plt.subplots(1, 1, figsize=(11, 4))
    pos_err_norm = np.linalg.norm(pos_err_hist, axis=1)
    vel_err_norm = np.linalg.norm(vel_err_hist, axis=1)
    ax3.plot(t_ctrl, pos_err_norm, "b-", linewidth=2, label="Position tracking error norm")
    ax3.plot(t_ctrl, vel_err_norm, "g-", linewidth=2, label="Velocity tracking error norm")
    ax3.set_title("State Tracking Error")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error norm")
    ax3.legend(loc="best")
    fig3.tight_layout()

    # 5) Control inputs
    fig4, (ax41, ax42, ax43) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax41.plot(t_ctrl, np.degrees(u_hist[:, 0]), "b-", linewidth=1.8, label="phi_cmd")
    ax42.plot(t_ctrl, np.degrees(u_hist[:, 1]), "g-", linewidth=1.8, label="theta_cmd")
    ax43.plot(t_ctrl, u_hist[:, 2], "r-", linewidth=1.8, label="a_cmd")
    ax41.set_ylabel("deg")
    ax42.set_ylabel("deg")
    ax43.set_ylabel("m/s^2")
    ax43.set_xlabel("Time (s)")
    ax41.set_title("Control Inputs")
    ax41.legend(loc="best")
    ax42.legend(loc="best")
    ax43.legend(loc="best")
    fig4.tight_layout()

    # 6) Tilt angles
    fig5, (ax51, ax52) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax51.plot(t_state, np.degrees(x_hist[:, 6]), "b-", linewidth=1.8, label="roll phi (actual)")
    ax52.plot(t_state, np.degrees(x_hist[:, 7]), "g-", linewidth=1.8, label="pitch theta (actual)")
    ax51.plot(t_ctrl, np.degrees(u_hist[:, 0]), "b--", alpha=0.7, label="roll phi_cmd")
    ax52.plot(t_ctrl, np.degrees(u_hist[:, 1]), "g--", alpha=0.7, label="pitch theta_cmd")
    ax51.set_ylabel("deg")
    ax52.set_ylabel("deg")
    ax52.set_xlabel("Time (s)")
    ax51.set_title("Tilt Angles")
    ax51.legend(loc="best")
    ax52.legend(loc="best")
    fig5.tight_layout()

    # 7) Velocity norm
    fig6, ax6 = plt.subplots(1, 1, figsize=(11, 4))
    vel_norm = np.linalg.norm(x_hist[:, 3:6], axis=1)
    ax6.plot(t_state, vel_norm, "m-", linewidth=2, label="||v||")
    ax6.axhline(v_max, color="k", linestyle="--", linewidth=1.0, label="V_MAX")
    ax6.set_title("Velocity Norm")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("m/s")
    ax6.legend(loc="best")
    fig6.tight_layout()

    # 8) Virtual control norm (very important)
    fig7, (ax71, ax72) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    if virtual_norm_history is not None and virtual_norm_history.size > 0:
        ax71.plot(t_ctrl[: virtual_norm_history.size], virtual_norm_history, "r-", linewidth=2, label="Outer-step virtual norm")
    ax71.set_title("Virtual Control Norm")
    ax71.set_ylabel("||nu||")
    ax71.legend(loc="best")
    if iter_virtual_norm_history is not None and iter_virtual_norm_history.size > 0:
        ax72.plot(np.arange(iter_virtual_norm_history.size), iter_virtual_norm_history, "k-", linewidth=1.8, label="Per SCP iteration")
    ax72.set_xlabel("SCP iteration index")
    ax72.set_ylabel("||nu||")
    ax72.legend(loc="best")
    fig7.tight_layout()

    # 9) Trust region evolution
    fig8, (ax81, ax82) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    if trust_history is not None and trust_history.size > 0:
        ax81.plot(t_ctrl[: trust_history.size], trust_history, "b-", linewidth=2, label="Outer-step trust radius")
    ax81.set_title("Trust Region Evolution")
    ax81.set_ylabel("radius")
    ax81.legend(loc="best")
    if iter_trust_history is not None and iter_trust_history.size > 0:
        ax82.plot(np.arange(iter_trust_history.size), iter_trust_history, "c-", linewidth=1.8, label="Per SCP iteration")
    ax82.set_xlabel("SCP iteration index")
    ax82.set_ylabel("radius")
    ax82.legend(loc="best")
    fig8.tight_layout()

    # 10) Cost convergence per SCP iteration
    fig9, (ax91, ax92) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    if cost_history is not None and cost_history.size > 0:
        ax91.plot(t_ctrl[: cost_history.size], cost_history, "m-", linewidth=2, label="Outer-step accepted cost")
    ax91.set_title("Cost Convergence")
    ax91.set_ylabel("Cost")
    ax91.legend(loc="best")
    if iter_cost_history is not None and iter_cost_history.size > 0:
        ax92.plot(np.arange(iter_cost_history.size), iter_cost_history, "k-", linewidth=1.8, label="Per SCP iteration")
    ax92.set_xlabel("SCP iteration index")
    ax92.set_ylabel("Cost")
    ax92.legend(loc="best")
    fig9.tight_layout()

    # 11) Constraint residuals (obstacle and cone)
    fig10, (ax101, ax102) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax101.plot(t_ctrl[: obstacle_residual_history.size], obstacle_residual_history, "r-", linewidth=2, label="Obstacle residual")
    ax101.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax101.set_ylabel("m")
    ax101.set_title("Constraint Residuals")
    ax101.legend(loc="best")

    ax102.plot(t_ctrl[: cone_residual_history.size], cone_residual_history, "c-", linewidth=2, label="Cone residual")
    ax102.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax102.set_ylabel("margin")
    ax102.set_xlabel("Time (s)")
    ax102.legend(loc="best")
    fig10.tight_layout()

    # 12) Phase-space plots
    fig11, (ax111, ax112, ax113) = plt.subplots(1, 3, figsize=(15, 4))
    ax111.plot(x_hist[:, 0], x_hist[:, 3], "b-", linewidth=1.8)
    ax112.plot(x_hist[:, 1], x_hist[:, 4], "g-", linewidth=1.8)
    ax113.plot(x_hist[:, 2], x_hist[:, 5], "m-", linewidth=1.8)
    ax111.set_xlabel("x (m)")
    ax111.set_ylabel("vx (m/s)")
    ax112.set_xlabel("y (m)")
    ax112.set_ylabel("vy (m/s)")
    ax113.set_xlabel("z (m)")
    ax113.set_ylabel("vz (m/s)")
    fig11.suptitle("Phase-Space Plots")
    fig11.tight_layout()

    # 13) Energy and time-to-go estimate
    fig12, (ax121, ax122) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    power_proxy = u_hist[:, 2] ** 2 + 0.5 * (u_hist[:, 0] ** 2 + u_hist[:, 1] ** 2)
    energy_proxy = np.cumsum(power_proxy) * dt
    rel_speed = np.linalg.norm(x_hist[:-1, 3:6] - target_vel_hist[:-1], axis=1)
    dist_ctrl = np.linalg.norm(x_hist[:-1, 0:3] - target_pos_hist[:-1], axis=1)
    tgo_est = dist_ctrl / np.maximum(rel_speed, 0.05)

    ax121.plot(t_ctrl, energy_proxy, "k-", linewidth=2, label="Cumulative energy proxy")
    ax121.set_ylabel("Energy proxy")
    ax121.set_title("Energy and Time-to-Go")
    ax121.legend(loc="best")

    ax122.plot(t_ctrl, tgo_est, "orange", linewidth=2, label="Time-to-go estimate")
    ax122.set_ylabel("s")
    ax122.set_xlabel("Time (s)")
    ax122.legend(loc="best")
    fig12.tight_layout()

    # Auxiliary: delta history (SCP contraction health)
    if delta_history is not None and delta_history.size > 0:
        fig13, ax13 = plt.subplots(1, 1, figsize=(11, 4))
        ax13.plot(t_ctrl[: delta_history.size], delta_history, "purple", linewidth=2, label="SCP delta (inf norm)")
        ax13.set_title("SCP Delta History")
        ax13.set_xlabel("Time (s)")
        ax13.set_ylabel("delta")
        ax13.legend(loc="best")
        fig13.tight_layout()

    plt.show()
