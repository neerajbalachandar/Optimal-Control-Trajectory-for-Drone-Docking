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
DEFAULT_SCP_TIME_SCALE = 1.0
DEFAULT_HOLD_SEC = 6.0
DEFAULT_MAX_EXTRA_SEC = 20.0

DEFAULT_DOCK_DIST_TOL = 0.12
DEFAULT_DOCK_VEL_TOL = 0.20

TRACK_POS_KP = np.array([1.2, 1.2, 1.8])
TRACK_POS_KD = np.array([1.0, 1.0, 1.4])
DOCK_POS_KP = np.array([1.8, 1.8, 2.6])
DOCK_POS_KD = np.array([1.2, 1.2, 1.8])

# Geometric attitude gains that generate desired angular acceleration (rad/s^2).
ATT_KR = np.array([2200.0, 2200.0, 1800.0])
ATT_KW = np.array([140.0, 140.0, 100.0])

TRACK_ACCEL_CLIP = 6.0


def load_scp_static_fullvec_solution(suppress_plots: bool = True):
    """Run theory/SCP_static_fullvec.py and extract converged x_nom and u_nom."""
    scp_path = Path(__file__).resolve().parents[1] / "theory" / "SCP_static_fullvec.py"
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
            f"Missing dependency {exc.name} required by {scp_path.name}. "
            "Install SCP_static_fullvec.py dependencies for example cvxpy and rerun."
        )
        raise RuntimeError(msg) from exc
    finally:
        plt.show = original_show

    required = [
        "x_nom",
        "u_nom",
        "dt",
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
        raise RuntimeError(f"SCP_static_fullvec.py did not expose expected outputs: {missing}")

    x_traj = np.asarray(data["x_nom"], dtype=float)
    u_traj = np.asarray(data["u_nom"], dtype=float)

    if x_traj.ndim != 2 or x_traj.shape[1] != 13:
        raise RuntimeError(f"Unexpected x_nom shape {x_traj.shape}; expected (N, 13).")
    if u_traj.shape != (x_traj.shape[0] - 1, 4):
        raise RuntimeError(
            f"Unexpected u_nom shape {u_traj.shape}; expected {(x_traj.shape[0] - 1, 4)}."
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


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    nrm = np.linalg.norm(q)
    if nrm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / nrm


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q_wxyz)
    return np.array([q[1], q[2], q[3], q[0]])


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    return np.array(p.getMatrixFromQuaternion(q_xyzw)).reshape(3, 3)


def yaw_from_rotmat(rot: np.ndarray) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


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
    """Disable collisions between the two drone multibodies.

    This allows center-to-center docking against a static target pose.
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


def allocate_cf2x_rpm(total_thrust_n: float, tau_cmd_nm: np.ndarray, env: CtrlAviary) -> np.ndarray:
    arm = env.L / np.sqrt(2.0)
    yaw_coeff = env.KM / env.KF

    alloc = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [arm, arm, -arm, -arm],
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


def scale_thrust_for_time_replay(thrust_n: float, time_scale: float, gravity_force: float) -> float:
    if time_scale <= 0.0:
        return thrust_n
    return gravity_force + (thrust_n - gravity_force) / (time_scale**2)


def stabilized_rpm_command(
    obs_chaser: np.ndarray,
    thrust_ff_n: float,
    tau_ff_nm: np.ndarray,
    q_des_wxyz: np.ndarray,
    omega_des_body: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_kp: np.ndarray,
    pos_kd: np.ndarray,
    env: CtrlAviary,
) -> np.ndarray:
    pos = obs_chaser[0:3]
    quat_xyzw = obs_chaser[3:7]
    vel_world = obs_chaser[10:13]
    omega_world = obs_chaser[13:16]

    rot = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    omega_body = rot.T @ omega_world

    rot_ff = quat_wxyz_to_rotmat(q_des_wxyz)
    b3_ff = rot_ff[:, 2]

    # Feedforward force from SCP and feedback correction from position and velocity tracking.
    force_ff_world = float(thrust_ff_n) * b3_ff
    accel_fb = pos_kp * (pos_ref - pos) + pos_kd * (vel_ref - vel_world)
    accel_fb = np.clip(accel_fb, -TRACK_ACCEL_CLIP, TRACK_ACCEL_CLIP)
    force_fb_world = env.M * accel_fb
    force_cmd_world = force_ff_world + force_fb_world

    if np.linalg.norm(force_cmd_world) < 1e-6:
        force_cmd_world = np.array([0.0, 0.0, env.GRAVITY])

    yaw_des = yaw_from_rotmat(rot_ff)
    rot_des = build_desired_rotation(force_cmd_world, yaw_des_rad=yaw_des)

    # Body thrust acts along the current body-z axis in PyBullet.
    thrust_cmd = float(np.dot(force_cmd_world, rot[:, 2]))
    thrust_cmd = np.clip(thrust_cmd, 0.0, env.MAX_THRUST)

    rot_err_mat = 0.5 * (rot_des.T @ rot - rot.T @ rot_des)
    e_rot = np.array([rot_err_mat[2, 1], rot_err_mat[0, 2], rot_err_mat[1, 0]])

    omega_des_body = np.asarray(omega_des_body, dtype=float)
    e_omega = omega_body - omega_des_body

    desired_ang_acc = -ATT_KR * e_rot - ATT_KW * e_omega
    tau_fb = env.J @ desired_ang_acc + np.cross(omega_body, env.J @ omega_body)
    tau_cmd = np.asarray(tau_ff_nm, dtype=float) + tau_fb
    tau_cmd[0:2] = np.clip(tau_cmd[0:2], -env.MAX_XY_TORQUE, env.MAX_XY_TORQUE)
    tau_cmd[2] = np.clip(tau_cmd[2], -env.MAX_Z_TORQUE, env.MAX_Z_TORQUE)

    return allocate_cf2x_rpm(thrust_cmd, tau_cmd, env)


def run(
    gui: bool = True,
    record_video: bool = False,
    sim_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    scp_time_scale: float = DEFAULT_SCP_TIME_SCALE,
    hold_sec: float = DEFAULT_HOLD_SEC,
    auto_extend: bool = True,
    max_extra_sec: float = DEFAULT_MAX_EXTRA_SEC,
    freeze_target: bool = True,
    disable_inter_drone_collision: bool = True,
    dock_dist_tol: float = DEFAULT_DOCK_DIST_TOL,
    dock_vel_tol: float = DEFAULT_DOCK_VEL_TOL,
    show_scp_plots: bool = False,
):
    scp = load_scp_static_fullvec_solution(suppress_plots=not show_scp_plots)
    x_traj = scp["x_traj"]
    u_traj = scp["u_traj"]

    if scp_time_scale <= 0.0:
        raise RuntimeError("scp_time_scale must be > 0.")
    if sim_freq_hz % control_freq_hz != 0:
        raise RuntimeError("sim_freq_hz must be an integer multiple of control_freq_hz.")

    dt_scp = scp["dt"]
    chaser_init = scp["x0"][0:3]
    target_pos = scp["p_target"].copy()
    q0_wxyz = normalize_quat_wxyz(scp["x0"][6:10])

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

    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        chaser_init,
        quat_wxyz_to_xyzw(q0_wxyz),
        physicsClientId=pyb_client,
    )
    p.resetBaseVelocity(
        env.DRONE_IDS[0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        physicsClientId=pyb_client,
    )

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
            u_cmd = sample_control_zoh(u_traj, plan_t, dt_scp)
            x_cmd = sample_state_linear(x_traj, plan_t, dt_scp)

            thrust_ff = scale_thrust_for_time_replay(
                thrust_n=float(u_cmd[0]),
                time_scale=scp_time_scale,
                gravity_force=float(env.GRAVITY),
            )
            tau_ff = np.asarray(u_cmd[1:4], dtype=float) / (scp_time_scale**2)

            pos_ref = x_cmd[0:3]
            vel_ref = x_cmd[3:6] / scp_time_scale
            q_des = x_cmd[6:10]
            omega_des = x_cmd[10:13] / scp_time_scale
            pos_kp = TRACK_POS_KP
            pos_kd = TRACK_POS_KD
        else:
            thrust_ff = float(env.GRAVITY)
            tau_ff = np.zeros(3)

            pos_ref = target_pos
            vel_ref = np.zeros(3)
            q_des = x_traj[-1, 6:10]
            omega_des = np.zeros(3)
            pos_kp = DOCK_POS_KP
            pos_kd = DOCK_POS_KD

        action[0] = stabilized_rpm_command(
            obs_chaser=chaser,
            thrust_ff_n=thrust_ff,
            tau_ff_nm=tau_ff,
            q_des_wxyz=q_des,
            omega_des_body=omega_des,
            pos_ref=pos_ref,
            vel_ref=vel_ref,
            pos_kp=pos_kp,
            pos_kd=pos_kd,
            env=env,
        )
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
            "Animate static target docking using coupled full-state SCP outputs from "
            "theory SCP_static_fullvec.py and a stabilizing low level attitude controller."
        )
    )
    parser.add_argument("--gui", default=True, type=str2bool, help="Use PyBullet GUI")
    parser.add_argument("--record_video", default=False, type=str2bool, help="Record PyBullet video")
    parser.add_argument(
        "--sim_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="PyBullet simulation frequency must be multiple of control frequency",
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
        help="Time scaling applied to SCP replay",
    )
    parser.add_argument(
        "--hold_sec",
        default=DEFAULT_HOLD_SEC,
        type=float,
        help="Extra time after SCP horizon",
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
        help="Maximum extension time when auto_extend is enabled",
    )
    parser.add_argument(
        "--freeze_target",
        default=True,
        type=str2bool,
        help="Keep target drone fixed at target position",
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
        help="Docking distance tolerance used by stop condition",
    )
    parser.add_argument(
        "--dock_vel_tol",
        default=DEFAULT_DOCK_VEL_TOL,
        type=float,
        help="Docking velocity tolerance used by stop condition",
    )
    parser.add_argument(
        "--show_scp_plots",
        default=False,
        type=str2bool,
        help="Allow SCP_static_fullvec.py plots to display",
    )
    run(**vars(parser.parse_args()))
