import argparse
import runpy
import time
from pathlib import Path

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync


DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_SCP_TIME_SCALE = 3.0 # Increasing does not help in guaranteeing the use of SCP generated control input to make the drone dock
DEFAULT_HOLD_SEC = 8.0
DEFAULT_MAX_EXTRA_SEC = 10.0

DEFAULT_DOCK_DIST_TOL = 0.15
DEFAULT_DOCK_VEL_TOL = 0.25

DEFAULT_YAW_DEG = 0.0
DEFAULT_YAW_RAD = np.deg2rad(DEFAULT_YAW_DEG)

# Optional tracking around SCP states in full rigid-body dynamics.
STATE_TRACK_KP = np.array([1.4, 1.4, 2.2])
STATE_TRACK_KD = np.array([1.0, 1.0, 1.6])
FEEDBACK_ACCEL_CLIP = 6.0
SCP_ACCEL_CLIP = 15.0

# Terminal docking controller once SCP horizon ends (NOT neutral hover).
HOVER_KP = np.array([1.8, 1.8, 2.6])
HOVER_KD = np.array([1.1, 1.1, 1.6])
DOCK_DESCENT_BIAS = 0.8

# Geometric attitude control gains (body-axis torque command, N*m).
K_R_NORM = np.array([2200.0, 2200.0, 1800.0])
K_W_NORM = np.array([140.0, 140.0, 100.0])

# Visual aids


def load_scp_static_solution(suppress_plots: bool = True):
    """Run theory/SCP_static.py and extract its converged trajectories."""
    scp_path = Path(__file__).resolve().parents[1] / "theory" / "SCP_static.py"
    if not scp_path.exists():
        raise FileNotFoundError(f"Could not find SCP file: {scp_path}")

    try:
        if suppress_plots:
            import matplotlib

            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency for loading SCP solution: {exc}") from exc

    original_show = plt.show
    if suppress_plots:
        plt.show = lambda *args, **kwargs: None

    try:
        data = runpy.run_path(str(scp_path))
    except ModuleNotFoundError as exc:
        msg = (
            f"Missing dependency '{exc.name}' required by {scp_path.name}. "
            "Install SCP_static.py dependencies (e.g., cvxpy and a supported solver) and rerun."
        )
        raise RuntimeError(msg) from exc
    finally:
        plt.show = original_show

    required = [
        "x_nom",
        "u_nom",
        "dt",
        "N",
        "x0",
        "p_target",
        "P_OBS",
        "R_OBS",
        "THETA",
        "N_APP",
        "r_c",
        "r_t",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"SCP_static.py did not expose expected outputs: {missing}")

    x_traj = np.asarray(data["x_nom"], dtype=float)
    u_traj = np.asarray(data["u_nom"], dtype=float)

    if x_traj.ndim != 2 or x_traj.shape[1] != 6:
        raise RuntimeError(f"Unexpected x_nom shape {x_traj.shape}; expected (N, 6).")
    if u_traj.shape != (x_traj.shape[0] - 1, 3):
        raise RuntimeError(
            f"Unexpected u_nom shape {u_traj.shape}; expected {(x_traj.shape[0] - 1, 3)}."
        )

    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "dt": float(data["dt"]),
        "x0": np.asarray(data["x0"], dtype=float),
        "p_target": np.asarray(data["p_target"], dtype=float),
        "p_obs": np.asarray(data["P_OBS"], dtype=float),
        "r_obs": float(data["R_OBS"]),
        "cone_half_angle_rad": float(data["THETA"]),
        "cone_axis": np.asarray(data["N_APP"], dtype=float),
        "r_c": float(data["r_c"]),
        "r_t": float(data["r_t"]),
    }


def draw_trajectory(traj_xyz: np.ndarray, client: int):
    for i in range(traj_xyz.shape[0] - 1):
        p.addUserDebugLine(
            traj_xyz[i],
            traj_xyz[i + 1],
            lineColorRGB=[0.1, 0.4, 1.0],
            lineWidth=2.0,
            physicsClientId=client,
        )


def draw_docking_cone(target_pos, axis_normal, cone_half_angle_rad, length, client):
    axis_normal = axis_normal / (np.linalg.norm(axis_normal) + 1e-9)
    cone_dir = -axis_normal

    if abs(axis_normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u_vec = np.cross(axis_normal, ref)
    u_vec /= np.linalg.norm(u_vec) + 1e-9
    v_vec = np.cross(axis_normal, u_vec)

    for phi in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        radial = u_vec * np.cos(phi) + v_vec * np.sin(phi)
        ray = cone_dir * np.cos(cone_half_angle_rad) + radial * np.sin(cone_half_angle_rad)
        end = target_pos + length * ray
        p.addUserDebugLine(target_pos, end, [0.0, 1.0, 0.0], 1.5, physicsClientId=client)

    p.addUserDebugLine(
        target_pos,
        target_pos + cone_dir * length,
        [0.0, 1.0, 0.0],
        3.0,
        physicsClientId=client,
    )


def add_visual_sphere(position, radius, rgba, client):
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )


def update_visual_sphere(body_id: int, position: np.ndarray, client: int):
    p.resetBasePositionAndOrientation(
        body_id,
        position,
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=client,
    )


def reset_camera(chaser_init: np.ndarray, target_pos: np.ndarray, client: int):
    center = 0.5 * (chaser_init + target_pos)
    span = np.linalg.norm(target_pos - chaser_init)
    camera_distance = max(2.5, span + 1.2)
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=35.0,
        cameraPitch=-24.0,
        cameraTargetPosition=center,
        physicsClientId=client,
    )


def freeze_target_drone(env: CtrlAviary, target_pos: np.ndarray):
    target_id = env.DRONE_IDS[1]
    p.resetBasePositionAndOrientation(
        target_id,
        target_pos,
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=env.CLIENT,
    )
    p.resetBaseVelocity(
        target_id,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        physicsClientId=env.CLIENT,
    )


def disable_collision_between_drones(env: CtrlAviary, chaser_idx: int = 0, target_idx: int = 1):
    """Disable collisions between chaser and static target bodies.

    This avoids physical contact locking the chaser above the target center.
    """
    body_a = env.DRONE_IDS[chaser_idx]
    body_b = env.DRONE_IDS[target_idx]
    joints_a = [-1] + list(range(p.getNumJoints(body_a, physicsClientId=env.CLIENT)))
    joints_b = [-1] + list(range(p.getNumJoints(body_b, physicsClientId=env.CLIENT)))
    for link_a in joints_a:
        for link_b in joints_b:
            p.setCollisionFilterPair(
                bodyUniqueIdA=body_a,
                bodyUniqueIdB=body_b,
                linkIndexA=link_a,
                linkIndexB=link_b,
                enableCollision=0,
                physicsClientId=env.CLIENT,
            )


def sample_state_linear(x_traj: np.ndarray, t: float, dt_scp: float) -> np.ndarray:
    if x_traj.shape[0] == 1:
        return x_traj[0].copy()

    max_t = (x_traj.shape[0] - 1) * dt_scp
    tau = float(np.clip(t, 0.0, max_t))
    idx0 = int(np.floor(tau / dt_scp))
    idx1 = min(idx0 + 1, x_traj.shape[0] - 1)

    if idx1 == idx0:
        return x_traj[idx0].copy()

    alpha = (tau - idx0 * dt_scp) / dt_scp
    return (1.0 - alpha) * x_traj[idx0] + alpha * x_traj[idx1]


def sample_control_zoh(u_traj: np.ndarray, t: float, dt_scp: float) -> np.ndarray:
    idx = int(np.floor(max(t, 0.0) / dt_scp))
    idx = min(max(idx, 0), u_traj.shape[0] - 1)
    return u_traj[idx].copy()


def build_desired_rotation(force_world: np.ndarray, yaw_des_rad: float) -> np.ndarray:
    z_body_des = force_world / (np.linalg.norm(force_world) + 1e-9)

    x_c = np.array([np.cos(yaw_des_rad), np.sin(yaw_des_rad), 0.0])
    y_body_des = np.cross(z_body_des, x_c)
    if np.linalg.norm(y_body_des) < 1e-6:
        x_c = np.array([1.0, 0.0, 0.0])
        y_body_des = np.cross(z_body_des, x_c)
    y_body_des /= np.linalg.norm(y_body_des) + 1e-9
    x_body_des = np.cross(y_body_des, z_body_des)
    x_body_des /= np.linalg.norm(x_body_des) + 1e-9

    return np.column_stack((x_body_des, y_body_des, z_body_des))


def allocate_cf2x_rpm(total_thrust_n: float, tau_cmd_nm: np.ndarray, env: CtrlAviary) -> np.ndarray:
    arm = env.L / np.sqrt(2.0)
    yaw_coeff = env.KM / env.KF

    alloc = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-yaw_coeff, yaw_coeff, -yaw_coeff, yaw_coeff],
        ]
    )
    desired = np.array([total_thrust_n, tau_cmd_nm[0], tau_cmd_nm[1], tau_cmd_nm[2]])

    motor_forces = np.linalg.solve(alloc, desired)
    motor_force_max = env.KF * env.MAX_RPM**2
    motor_forces = np.clip(motor_forces, 0.0, motor_force_max)
    rpm = np.sqrt(motor_forces / env.KF)
    return np.clip(rpm, 0.0, env.MAX_RPM)


def accel_to_rpm(accel_des_world: np.ndarray, obs: np.ndarray, env: CtrlAviary, yaw_des_rad: float) -> np.ndarray:
    """Map desired inertial acceleration to motor RPMs using rigid-body allocation."""
    quat = obs[3:7]
    omega_world = obs[13:16]
    rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    omega_body = rot.T @ omega_world

    # m*v_dot = R*[0,0,f]^T - m*g*e3
    g_acc = env.GRAVITY / env.M
    force_world_des = env.M * (accel_des_world + np.array([0.0, 0.0, g_acc]))
    rot_des = build_desired_rotation(force_world_des, yaw_des_rad=yaw_des_rad)

    total_thrust_n = float(np.dot(force_world_des, rot[:, 2]))
    total_thrust_n = np.clip(total_thrust_n, 0.0, env.MAX_THRUST)

    rot_err_mat = 0.5 * (rot_des.T @ rot - rot.T @ rot_des)
    e_rot = np.array([rot_err_mat[2, 1], rot_err_mat[0, 2], rot_err_mat[1, 0]])

    desired_ang_acc = -K_R_NORM * e_rot - K_W_NORM * omega_body
    tau_cmd = env.J @ desired_ang_acc + np.cross(omega_body, env.J @ omega_body)
    tau_cmd[0:2] = np.clip(tau_cmd[0:2], -env.MAX_XY_TORQUE, env.MAX_XY_TORQUE)
    tau_cmd[2] = np.clip(tau_cmd[2], -env.MAX_Z_TORQUE, env.MAX_Z_TORQUE)

    return allocate_cf2x_rpm(total_thrust_n, tau_cmd, env)


def run(
    gui: bool = True,
    record_video: bool = False,
    sim_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    scp_time_scale: float = DEFAULT_SCP_TIME_SCALE,
    hold_sec: float = DEFAULT_HOLD_SEC,
    auto_extend: bool = True,
    max_extra_sec: float = DEFAULT_MAX_EXTRA_SEC,
    use_state_feedback: bool = True,
    freeze_target: bool = True,
    disable_inter_drone_collision: bool = True,
    dock_dist_tol: float = DEFAULT_DOCK_DIST_TOL,
    dock_vel_tol: float = DEFAULT_DOCK_VEL_TOL,
    show_scp_plots: bool = False,
):
    scp = load_scp_static_solution(suppress_plots=not show_scp_plots)
    x_traj = scp["x_traj"]
    u_traj = scp["u_traj"]

    if scp_time_scale <= 0.0:
        raise RuntimeError("scp_time_scale must be > 0.")
    if sim_freq_hz % control_freq_hz != 0:
        raise RuntimeError("sim_freq_hz must be an integer multiple of control_freq_hz.")

    dt_scp = scp["dt"]
    chaser_init = scp["x0"][0:3]
    target_pos = scp["p_target"].copy()
    dock_axis = scp["cone_axis"] / (np.linalg.norm(scp["cone_axis"]) + 1e-9)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([chaser_init, target_pos]),
        initial_rpys=np.zeros((2, 3)),
        physics=Physics.PYB,
        neighbourhood_radius=np.inf,
        pyb_freq=sim_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=False,
        user_debug_gui=False,
    )
    pyb_client = env.getPyBulletClient()

    if gui:
        reset_camera(chaser_init, target_pos, pyb_client)

    add_visual_sphere(scp["p_obs"], scp["r_obs"], [1.0, 0.2, 0.2, 0.35], pyb_client)
    draw_docking_cone(
        target_pos=target_pos,
        axis_normal=scp["cone_axis"],
        cone_half_angle_rad=scp["cone_half_angle_rad"],
        length=1.2,
        client=pyb_client,
    )
    draw_trajectory(x_traj[:, 0:3], pyb_client)

    chaser_hull = add_visual_sphere(chaser_init, scp["r_c"], [0.0, 1.0, 1.0, 0.30], pyb_client)
    target_hull = add_visual_sphere(target_pos, scp["r_t"], [1.0, 0.0, 1.0, 0.30], pyb_client)

    action = np.zeros((2, 4), dtype=float)
    action[0] = np.full(4, env.HOVER_RPM)
    action[1] = np.zeros(4)

    if disable_inter_drone_collision:
        disable_collision_between_drones(env, chaser_idx=0, target_idx=1)

    if freeze_target:
        freeze_target_drone(env, target_pos)

    duration_scp = u_traj.shape[0] * dt_scp
    duration_scaled = duration_scp * scp_time_scale
    base_duration = duration_scaled + max(hold_sec, 0.0)
    base_steps = int(np.ceil(base_duration * env.CTRL_FREQ))
    max_steps = base_steps + int(np.ceil(max(max_extra_sec, 0.0) * env.CTRL_FREQ))

    print(
        "[INFO] SCP horizon: "
        f"{duration_scp:.2f}s, replay scale: {scp_time_scale:.2f}x, "
        f"planned replay duration: {duration_scaled:.2f}s, "
        f"base sim duration: {base_duration:.2f}s"
    )

    start_wall = time.time()
    obs = None
    docked = False
    i = 0

    while i < max_steps:
        obs, _, _, _, _ = env.step(action)

        if freeze_target:
            freeze_target_drone(env, target_pos)

        chaser = obs[0]
        chaser_pos = chaser[0:3]
        chaser_vel = chaser[10:13]

        sim_t = i * env.CTRL_TIMESTEP
        if sim_t < duration_scaled:
            plan_t = sim_t / scp_time_scale
            accel_cmd = sample_control_zoh(u_traj, plan_t, dt_scp) / (scp_time_scale**2)
            if use_state_feedback:
                ref_t = min(plan_t + env.CTRL_TIMESTEP / scp_time_scale, (x_traj.shape[0] - 1) * dt_scp)
                x_ref = sample_state_linear(x_traj, ref_t, dt_scp)
                pos_ref = x_ref[0:3]
                vel_ref = x_ref[3:6] / scp_time_scale
                feedback_acc = STATE_TRACK_KP * (pos_ref - chaser_pos) + STATE_TRACK_KD * (vel_ref - chaser_vel)
                feedback_acc = np.clip(feedback_acc, -FEEDBACK_ACCEL_CLIP, FEEDBACK_ACCEL_CLIP)
                accel_cmd += feedback_acc
        else:
            rel_to_target = target_pos - chaser_pos
            accel_cmd = HOVER_KP * rel_to_target - HOVER_KD * chaser_vel

            # Phase-2 docking bias: keep descending along cone axis until final capture.
            axial_gap = float(np.dot(rel_to_target, dock_axis))
            if axial_gap > dock_dist_tol:
                accel_cmd += DOCK_DESCENT_BIAS * dock_axis

        accel_cmd = np.clip(accel_cmd, -SCP_ACCEL_CLIP, SCP_ACCEL_CLIP)
        action[0] = accel_to_rpm(accel_cmd, chaser, env, yaw_des_rad=DEFAULT_YAW_RAD)
        action[1] = np.zeros(4)

        update_visual_sphere(chaser_hull, chaser_pos, pyb_client)
        update_visual_sphere(target_hull, target_pos, pyb_client)

        rel_pos_norm = float(np.linalg.norm(chaser_pos - target_pos))
        rel_vel_norm = float(np.linalg.norm(chaser_vel))
        if sim_t >= duration_scaled and rel_pos_norm <= dock_dist_tol and rel_vel_norm <= dock_vel_tol:
            docked = True

        env.render()
        if gui:
            sync(i, start_wall, env.CTRL_TIMESTEP)

        i += 1

        if i >= base_steps:
            if not auto_extend:
                break
            if docked:
                break

    if obs is not None:
        rel_pos = obs[0][0:3] - target_pos
        rel_vel = obs[0][10:13]
        print(f"[RESULT] Docked: {docked}")
        print(f"[RESULT] Simulated time (s): {i * env.CTRL_TIMESTEP:.2f}")
        print(f"[RESULT] Final relative position (m): {rel_pos}")
        print(f"[RESULT] Final relative velocity (m/s): {rel_vel}")
        print(f"[RESULT] Final docking distance (m): {np.linalg.norm(rel_pos):.4f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Animate static-target docking by using x_nom/u_nom directly from theory/SCP_static.py "
            "and mapping desired inertial accelerations to motor RPMs with rigid-body dynamics."
        )
    )
    parser.add_argument("--gui", default=True, type=str2bool, help="Use PyBullet GUI")
    parser.add_argument("--record_video", default=False, type=str2bool, help="Record PyBullet video")
    parser.add_argument(
        "--sim_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="PyBullet simulation frequency (must be multiple of control frequency)",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
        help="Control loop frequency for replaying SCP commands",
    )
    parser.add_argument(
        "--scp_time_scale",
        default=DEFAULT_SCP_TIME_SCALE,
        type=float,
        help="Time scaling applied to SCP replay ( >1 slows down replay for feasibility)",
    )
    parser.add_argument(
        "--hold_sec",
        default=DEFAULT_HOLD_SEC,
        type=float,
        help="Extra hover time after SCP horizon",
    )
    parser.add_argument(
        "--auto_extend",
        default=True,
        type=str2bool,
        help="Extend simulation after base duration until docking tolerance is met",
    )
    parser.add_argument(
        "--max_extra_sec",
        default=DEFAULT_MAX_EXTRA_SEC,
        type=float,
        help="Maximum extra extension time when auto_extend is enabled",
    )
    parser.add_argument(
        "--use_state_feedback",
        default=True,
        type=str2bool,
        help="Add state-tracking correction around SCP u_nom using x_nom references",
    )
    parser.add_argument(
        "--freeze_target",
        default=True,
        type=str2bool,
        help="Keep target drone fixed at target position (no target dynamics)",
    )
    parser.add_argument(
        "--disable_inter_drone_collision",
        default=True,
        type=str2bool,
        help="Disable collisions between chaser and static target to allow center docking",
    )
    parser.add_argument(
        "--dock_dist_tol",
        default=DEFAULT_DOCK_DIST_TOL,
        type=float,
        help="Docking distance tolerance used by auto-extend stop condition",
    )
    parser.add_argument(
        "--dock_vel_tol",
        default=DEFAULT_DOCK_VEL_TOL,
        type=float,
        help="Docking velocity tolerance used by auto-extend stop condition",
    )
    parser.add_argument(
        "--show_scp_plots",
        default=False,
        type=str2bool,
        help="Allow SCP_static.py plots to display instead of suppressing them",
    )
    run(**vars(parser.parse_args()))
