import argparse
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync


DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_HOLD_SEC = 0.0

DEFAULT_YAW_DEG = 0.0
DEFAULT_YAW_RAD = np.deg2rad(DEFAULT_YAW_DEG)

SCP_ACCEL_CLIP = 15.0

# Geometric attitude control gains (body-axis torque command, N*m).
K_R_NORM = np.array([1200.0, 1200.0, 900.0])
K_W_NORM = np.array([80.0, 80.0, 60.0])


def load_scp_linear_ekf_solution(suppress_plots: bool = True):
    """Run theory/SCP_linear_EKF.py and extract executed trajectories."""
    scp_path = Path(__file__).resolve().parents[1] / "theory" / "SCP_linear_EKF.py"
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
            "Install SCP_linear_EKF.py dependencies (e.g., cvxpy and a supported solver) and rerun."
        )
        raise RuntimeError(msg) from exc
    finally:
        plt.show = original_show

    required = [
        "x_hist",
        "u_hist",
        "tar_hist",
        "tar_est_hist",
        "dt",
        "A_d",
        "B_d",
        "P_OBS",
        "R_OBS",
        "THETA",
        "N_APP",
        "r_c",
        "r_t",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"SCP_linear_EKF.py did not expose expected outputs: {missing}")

    x_exec = np.asarray(data["x_hist"], dtype=float)
    u_exec = np.asarray(data["u_hist"], dtype=float)
    tar_true = np.asarray(data["tar_hist"], dtype=float)
    tar_est = np.asarray(data["tar_est_hist"], dtype=float)
    A_d = np.asarray(data["A_d"], dtype=float)
    B_d = np.asarray(data["B_d"], dtype=float)

    if x_exec.ndim != 2 or x_exec.shape[1] != 6:
        raise RuntimeError(f"Unexpected x_hist shape {x_exec.shape}; expected (M, 6).")
    if u_exec.ndim != 2 or u_exec.shape[1] != 3:
        raise RuntimeError(f"Unexpected u_hist shape {u_exec.shape}; expected (K, 3).")
    if tar_true.ndim != 2 or tar_true.shape[1] != 6:
        raise RuntimeError(f"Unexpected tar_hist shape {tar_true.shape}; expected (M, 6).")
    if tar_est.ndim != 2 or tar_est.shape[1] != 6:
        raise RuntimeError(f"Unexpected tar_est_hist shape {tar_est.shape}; expected (M, 6).")
    if u_exec.shape[0] < 1:
        raise RuntimeError("SCP_linear_EKF.py produced no control sequence (u_hist is empty).")

    # Reconstruct initial chaser and target states from the first logged step.
    # In SCP_linear_EKF.py, histories are appended after propagation, so we prepend x(0).
    x0_chaser = np.linalg.solve(A_d, x_exec[0] - B_d @ u_exec[0])
    x0_target = np.linalg.solve(A_d, tar_true[0])

    x_exec = np.vstack([x0_chaser, x_exec])
    tar_true = np.vstack([x0_target, tar_true])

    # Keep estimated target history length aligned for interpolation convenience.
    if tar_est.shape[0] == u_exec.shape[0]:
        tar_est0 = np.linalg.solve(A_d, tar_est[0])
        tar_est = np.vstack([tar_est0, tar_est])

    return {
        "x_exec": x_exec,
        "u_exec": u_exec,
        "tar_true": tar_true,
        "tar_est": tar_est,
        "dt": float(data["dt"]),
        "p_obs": np.asarray(data["P_OBS"], dtype=float),
        "r_obs": float(data["R_OBS"]),
        "cone_half_angle_rad": float(data["THETA"]),
        "cone_axis": np.asarray(data["N_APP"], dtype=float),
        "r_c": float(data["r_c"]),
        "r_t": float(data["r_t"]),
    }


def draw_trajectory(traj_xyz: np.ndarray, client: int, color=(0.1, 0.4, 1.0), width=2.0):
    for i in range(traj_xyz.shape[0] - 1):
        p.addUserDebugLine(
            traj_xyz[i],
            traj_xyz[i + 1],
            lineColorRGB=list(color),
            lineWidth=width,
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
    camera_distance = max(3.5, span + 2.0)
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=35.0,
        cameraPitch=-24.0,
        cameraTargetPosition=center,
        physicsClientId=client,
    )


def set_target_drone_state(env: CtrlAviary, target_state: np.ndarray):
    target_id = env.DRONE_IDS[1]
    p.resetBasePositionAndOrientation(
        target_id,
        target_state[0:3],
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=env.CLIENT,
    )
    p.resetBaseVelocity(
        target_id,
        target_state[3:6],
        [0.0, 0.0, 0.0],
        physicsClientId=env.CLIENT,
    )


def disable_collision_between_drones(env: CtrlAviary, chaser_idx: int = 0, target_idx: int = 1):
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


def sample_state_linear(x_traj: np.ndarray, t: float, dt_plan: float) -> np.ndarray:
    if x_traj.shape[0] == 1:
        return x_traj[0].copy()

    max_t = (x_traj.shape[0] - 1) * dt_plan
    tau = float(np.clip(t, 0.0, max_t))
    idx0 = int(np.floor(tau / dt_plan))
    idx1 = min(idx0 + 1, x_traj.shape[0] - 1)

    if idx1 == idx0:
        return x_traj[idx0].copy()

    alpha = (tau - idx0 * dt_plan) / dt_plan
    return (1.0 - alpha) * x_traj[idx0] + alpha * x_traj[idx1]


def sample_control_zoh(u_traj: np.ndarray, t: float, dt_plan: float) -> np.ndarray:
    idx = int(np.floor(max(t, 0.0) / dt_plan))
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
    """Map desired inertial acceleration to motor RPMs with feasibility guards."""
    quat = obs[3:7]
    omega_world = obs[13:16]
    rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    omega_body = rot.T @ omega_world

    g_acc = env.GRAVITY / env.M
    accel_cmd = np.asarray(accel_des_world, dtype=float).copy()

    # Rotorcraft cannot command sustained downward acceleration beyond free-fall.
    accel_cmd[2] = max(accel_cmd[2], -0.95 * g_acc)

    force_world_des = env.M * (accel_cmd + np.array([0.0, 0.0, g_acc]))

    # Keep desired force inside motor authority while preserving direction.
    force_norm = np.linalg.norm(force_world_des)
    if force_norm > env.MAX_THRUST:
        force_world_des *= env.MAX_THRUST / force_norm
    elif force_norm < 1e-9:
        force_world_des = np.array([0.0, 0.0, 1e-6])

    rot_des = build_desired_rotation(force_world_des, yaw_des_rad=yaw_des_rad)

    # Project desired force onto current body-z axis for physically consistent thrust.
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
    hold_sec: float = DEFAULT_HOLD_SEC,
    disable_inter_drone_collision: bool = True,
    show_scp_plots: bool = False,
):
    scp = load_scp_linear_ekf_solution(suppress_plots=not show_scp_plots)
    x_exec = scp["x_exec"]
    u_exec = scp["u_exec"]
    tar_true = scp["tar_true"]
    tar_est = scp["tar_est"]

    if sim_freq_hz % control_freq_hz != 0:
        raise RuntimeError("sim_freq_hz must be an integer multiple of control_freq_hz.")

    dt_plan = scp["dt"]
    chaser_init = x_exec[0, 0:3]
    target_init = tar_true[0, 0:3]

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([chaser_init, target_init]),
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
    obs, _ = env.reset()

    if gui:
        reset_camera(chaser_init, target_init, pyb_client)

    add_visual_sphere(scp["p_obs"], scp["r_obs"], [1.0, 0.2, 0.2, 0.35], pyb_client)
    draw_trajectory(x_exec[:, 0:3], pyb_client, color=(0.1, 0.4, 1.0), width=2.0)
    draw_trajectory(tar_true[:, 0:3], pyb_client, color=(1.0, 0.1, 0.1), width=2.0)
    draw_trajectory(tar_est[:, 0:3], pyb_client, color=(1.0, 0.9, 0.1), width=1.5)
    draw_docking_cone(
        target_pos=target_init,
        axis_normal=scp["cone_axis"],
        cone_half_angle_rad=scp["cone_half_angle_rad"],
        length=1.2,
        client=pyb_client,
    )

    chaser_hull = add_visual_sphere(chaser_init, scp["r_c"], [0.0, 1.0, 1.0, 0.30], pyb_client)
    target_hull = add_visual_sphere(target_init, scp["r_t"], [1.0, 0.0, 1.0, 0.30], pyb_client)

    action = np.zeros((2, 4), dtype=float)
    action[0] = np.full(4, env.HOVER_RPM)
    action[1] = np.full(4, env.HOVER_RPM)

    if disable_inter_drone_collision:
        disable_collision_between_drones(env, chaser_idx=0, target_idx=1)

    set_target_drone_state(env, tar_true[0])

    duration_cmd = u_exec.shape[0] * dt_plan
    total_duration = duration_cmd + max(hold_sec, 0.0)
    max_steps = int(np.ceil(total_duration * env.CTRL_FREQ))

    print(
        "[INFO] Loaded SCP_linear_EKF replay: "
        f"{u_exec.shape[0]} controls @ dt={dt_plan:.3f}s, "
        f"command horizon={duration_cmd:.2f}s, total duration={total_duration:.2f}s"
    )

    pos_track_err = []
    vel_track_err = []

    start_wall = time.time()
    i = 0
    while i < max_steps:
        sim_t = i * env.CTRL_TIMESTEP
        target_state_now = sample_state_linear(tar_true, sim_t, dt_plan)
        set_target_drone_state(env, target_state_now)

        chaser = obs[0]
        if sim_t < duration_cmd:
            accel_cmd = sample_control_zoh(u_exec, sim_t, dt_plan)
        else:
            accel_cmd = np.zeros(3)

        accel_cmd = np.clip(accel_cmd, -SCP_ACCEL_CLIP, SCP_ACCEL_CLIP)
        action[0] = accel_to_rpm(accel_cmd, chaser, env, yaw_des_rad=DEFAULT_YAW_RAD)
        action[1] = np.full(4, env.HOVER_RPM)

        obs, _, _, _, _ = env.step(action)

        sim_t_next = (i + 1) * env.CTRL_TIMESTEP
        target_state_next = sample_state_linear(tar_true, sim_t_next, dt_plan)
        set_target_drone_state(env, target_state_next)

        chaser_pos = obs[0][0:3]
        chaser_vel = obs[0][10:13]
        update_visual_sphere(chaser_hull, chaser_pos, pyb_client)
        update_visual_sphere(target_hull, target_state_next[0:3], pyb_client)

        x_ref = sample_state_linear(x_exec, sim_t, dt_plan)
        pos_track_err.append(float(np.linalg.norm(chaser_pos - x_ref[0:3])))
        vel_track_err.append(float(np.linalg.norm(chaser_vel - x_ref[3:6])))

        if gui:
            env.render()
            sync(i, start_wall, env.CTRL_TIMESTEP)
        i += 1

    target_final = sample_state_linear(tar_true, i * env.CTRL_TIMESTEP, dt_plan)
    rel_pos = obs[0][0:3] - target_final[0:3]
    rel_vel = obs[0][10:13] - target_final[3:6]
    print(f"[RESULT] Simulated time (s): {i * env.CTRL_TIMESTEP:.2f}")
    print(f"[RESULT] Final relative position (m): {rel_pos}")
    print(f"[RESULT] Final relative velocity (m/s): {rel_vel}")
    print(f"[RESULT] Final docking distance (m): {np.linalg.norm(rel_pos):.4f}")
    print(f"[RESULT] Mean position tracking error vs SCP replay (m): {np.mean(pos_track_err):.4f}")
    print(f"[RESULT] Mean velocity tracking error vs SCP replay (m/s): {np.mean(vel_track_err):.4f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Animate SCP_linear_EKF.py by replaying its online-updated control history u_hist "
            "without translational PD; only attitude control is used in accel-to-RPM mapping."
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
        help="Control loop frequency for replaying SCP controls",
    )
    parser.add_argument(
        "--hold_sec",
        default=DEFAULT_HOLD_SEC,
        type=float,
        help="Extra time after control horizon with zero translational acceleration",
    )
    parser.add_argument(
        "--disable_inter_drone_collision",
        default=True,
        type=str2bool,
        help="Disable collisions between chaser and target",
    )
    parser.add_argument(
        "--show_scp_plots",
        default=False,
        type=str2bool,
        help="Allow SCP_linear_EKF.py plots to display instead of suppressing them",
    )
    run(**vars(parser.parse_args()))
