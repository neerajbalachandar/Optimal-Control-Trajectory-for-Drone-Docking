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
DEFAULT_PHYSICS_MODE = "pyb"

DEFAULT_YAW_DEG = 0.0
DEFAULT_YAW_RAD = np.deg2rad(DEFAULT_YAW_DEG)
DEFAULT_REPLAY_MODE = "theory_kinematic"

# Match the first-order actuator dynamics used in SCP_static_drone_dyn.py.
TAU_RP = 0.1
TAU_T = 0.05

# Geometric attitude control gains (body-axis torque command, N*m).
K_R_NORM = np.array([1200.0, 1200.0, 900.0])
K_W_NORM = np.array([80.0, 80.0, 60.0])


def load_scp_static_drone_dyn_solution(suppress_plots: bool = True, scp_seed: int | None = 0):
    """Run theory/SCP_static_drone_dyn.py and extract executed trajectories."""
    scp_path = Path(__file__).resolve().parents[1] / "theory" / "SCP_static_drone_dyn.py"
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

    rng_state = np.random.get_state()
    if scp_seed is not None:
        np.random.seed(int(scp_seed))

    try:
        data = runpy.run_path(str(scp_path))
    except ModuleNotFoundError as exc:
        msg = (
            f"Missing dependency '{exc.name}' required by {scp_path.name}. "
            "Install SCP_static_drone_dyn.py dependencies (e.g., cvxpy and a supported solver) and rerun."
        )
        raise RuntimeError(msg) from exc
    finally:
        plt.show = original_show
        np.random.set_state(rng_state)

    required = [
        "x_hist",
        "u_hist",
        "dt",
        "x0",
        "p_target_true",
        "P_OBS",
        "R_OBS",
        "THETA",
        "N_APP",
        "r_dock",
        "GRAVITY",
        "MAX_TILT",
        "U_MIN",
        "U_MAX",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"SCP_static_drone_dyn.py did not expose expected outputs: {missing}")

    x_exec = np.asarray(data["x_hist"], dtype=float)
    u_exec = np.asarray(data["u_hist"], dtype=float)

    if x_exec.ndim != 2 or x_exec.shape[1] != 9:
        raise RuntimeError(f"Unexpected x_hist shape {x_exec.shape}; expected (M, 9).")
    if u_exec.ndim != 2 or u_exec.shape[1] != 3:
        raise RuntimeError(f"Unexpected u_hist shape {u_exec.shape}; expected (K, 3).")

    return {
        "x_exec": x_exec,
        "u_exec": u_exec,
        "dt": float(data["dt"]),
        "x0": np.asarray(data["x0"], dtype=float),
        "target_pos": np.asarray(data["p_target_true"], dtype=float),
        "p_obs": np.asarray(data["P_OBS"], dtype=float),
        "r_obs": float(data["R_OBS"]),
        "cone_half_angle_rad": float(data["THETA"]),
        "cone_axis": np.asarray(data["N_APP"], dtype=float),
        "r_dock": float(data["r_dock"]),
        "gravity": float(data["GRAVITY"]),
        "max_tilt": float(data["MAX_TILT"]),
        "u_min": float(data["U_MIN"]),
        "u_max": float(data["U_MAX"]),
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


def build_desired_rotation_from_roll_pitch(roll_des: float, pitch_des: float, yaw_des: float) -> np.ndarray:
    quat_des = p.getQuaternionFromEuler([roll_des, pitch_des, yaw_des])
    return np.array(p.getMatrixFromQuaternion(quat_des)).reshape(3, 3)


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


def nmpc_cmd_to_rpm(
    u_cmd: np.ndarray,
    obs: np.ndarray,
    env: CtrlAviary,
    max_tilt_rad: float,
    yaw_des_rad: float,
) -> np.ndarray:
    """Map NMPC command [roll_cmd, pitch_cmd, thrust_accel_cmd] to motor RPM."""
    roll_cmd = float(np.clip(u_cmd[0], -max_tilt_rad, max_tilt_rad))
    pitch_cmd = float(np.clip(u_cmd[1], -max_tilt_rad, max_tilt_rad))
    thrust_acc_cmd = float(np.clip(u_cmd[2], 0.0, 1.5 * env.MAX_THRUST / env.M))

    quat = obs[3:7]
    omega_world = obs[13:16]
    rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    omega_body = rot.T @ omega_world

    rot_des = build_desired_rotation_from_roll_pitch(roll_cmd, pitch_cmd, yaw_des_rad)

    total_thrust_n = np.clip(env.M * thrust_acc_cmd, 0.0, env.MAX_THRUST)

    rot_err_mat = 0.5 * (rot_des.T @ rot - rot.T @ rot_des)
    e_rot = np.array([rot_err_mat[2, 1], rot_err_mat[0, 2], rot_err_mat[1, 0]])

    desired_ang_acc = -K_R_NORM * e_rot - K_W_NORM * omega_body
    tau_cmd = env.J @ desired_ang_acc + np.cross(omega_body, env.J @ omega_body)
    tau_cmd[0:2] = np.clip(tau_cmd[0:2], -env.MAX_XY_TORQUE, env.MAX_XY_TORQUE)
    tau_cmd[2] = np.clip(tau_cmd[2], -env.MAX_Z_TORQUE, env.MAX_Z_TORQUE)

    return allocate_cf2x_rpm(total_thrust_n, tau_cmd, env)


def apply_theory_actuator_lag(
    cmd_state: np.ndarray,
    cmd_target: np.ndarray,
    dt_ctrl: float,
    max_tilt_rad: float,
    u_min: float,
    u_max: float,
) -> np.ndarray:
    """Propagate [roll, pitch, thrust_acc] with the same lag model as theory dynamics."""
    next_state = cmd_state.copy()
    next_state[0] += dt_ctrl * (cmd_target[0] - cmd_state[0]) / TAU_RP
    next_state[1] += dt_ctrl * (cmd_target[1] - cmd_state[1]) / TAU_RP
    next_state[2] += dt_ctrl * (cmd_target[2] - cmd_state[2]) / TAU_T
    next_state[0:2] = np.clip(next_state[0:2], -max_tilt_rad, max_tilt_rad)
    next_state[2] = np.clip(next_state[2], u_min, u_max)
    return next_state


def set_chaser_from_reference(env: CtrlAviary, x_ref: np.ndarray, yaw_des_rad: float):
    """Hard-set chaser kinematics from theory reference state (kinematic replay mode)."""
    chaser_id = env.DRONE_IDS[0]
    quat_ref = p.getQuaternionFromEuler([x_ref[6], x_ref[7], yaw_des_rad])
    p.resetBasePositionAndOrientation(
        chaser_id,
        x_ref[0:3],
        quat_ref,
        physicsClientId=env.CLIENT,
    )
    p.resetBaseVelocity(
        chaser_id,
        x_ref[3:6],
        [0.0, 0.0, 0.0],
        physicsClientId=env.CLIENT,
    )


def run(
    gui: bool = True,
    record_video: bool = False,
    sim_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    hold_sec: float = DEFAULT_HOLD_SEC,
    replay_mode: str = DEFAULT_REPLAY_MODE,
    physics_mode: str = DEFAULT_PHYSICS_MODE,
    scp_seed: int = 0,
    freeze_target: bool = True,
    disable_inter_drone_collision: bool = True,
    show_scp_plots: bool = False,
):
    seed_for_scp = None if scp_seed < 0 else scp_seed
    scp = load_scp_static_drone_dyn_solution(
        suppress_plots=not show_scp_plots, scp_seed=seed_for_scp
    )
    x_exec = scp["x_exec"]
    u_exec = scp["u_exec"]

    if u_exec.shape[0] < 1:
        raise RuntimeError("SCP_static_drone_dyn.py produced no control sequence (u_hist is empty).")
    if sim_freq_hz % control_freq_hz != 0:
        raise RuntimeError("sim_freq_hz must be an integer multiple of control_freq_hz.")
    replay_mode = replay_mode.strip().lower()
    valid_modes = {"theory_kinematic", "pybullet_dynamic"}
    if replay_mode not in valid_modes:
        raise RuntimeError(f"Invalid replay_mode='{replay_mode}'. Expected one of {sorted(valid_modes)}.")
    physics_mode = physics_mode.strip().lower()
    physics_map = {"pyb": Physics.PYB, "dyn": Physics.DYN}
    if physics_mode not in physics_map:
        raise RuntimeError(f"Invalid physics_mode='{physics_mode}'. Expected one of {sorted(physics_map)}.")

    dt_plan = scp["dt"]
    chaser_init = scp["x0"][0:3]
    target_pos = scp["target_pos"].copy()

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=2,
        initial_xyzs=np.array([chaser_init, target_pos]),
        initial_rpys=np.zeros((2, 3)),
        physics=physics_map[physics_mode],
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
        reset_camera(chaser_init, target_pos, pyb_client)

    add_visual_sphere(scp["p_obs"], scp["r_obs"], [1.0, 0.2, 0.2, 0.35], pyb_client)
    draw_docking_cone(
        target_pos=target_pos,
        axis_normal=scp["cone_axis"],
        cone_half_angle_rad=scp["cone_half_angle_rad"],
        length=1.2,
        client=pyb_client,
    )
    draw_trajectory(x_exec[:, 0:3], pyb_client, color=(0.1, 0.4, 1.0), width=2.0)

    chaser_hull = add_visual_sphere(chaser_init, 0.1, [0.0, 1.0, 1.0, 0.30], pyb_client)
    target_hull = add_visual_sphere(target_pos, 0.1, [1.0, 0.0, 1.0, 0.30], pyb_client)

    action = np.zeros((2, 4), dtype=float)
    action[0] = np.full(4, env.HOVER_RPM)
    action[1] = np.full(4, env.HOVER_RPM)

    if disable_inter_drone_collision:
        disable_collision_between_drones(env, chaser_idx=0, target_idx=1)

    if freeze_target:
        freeze_target_drone(env, target_pos)
    if replay_mode == "theory_kinematic":
        set_chaser_from_reference(env, x_exec[0], yaw_des_rad=DEFAULT_YAW_RAD)
        obs = env._computeObs()

    duration_cmd = u_exec.shape[0] * dt_plan
    total_duration = duration_cmd + max(hold_sec, 0.0)
    max_steps = int(np.ceil(total_duration * env.CTRL_FREQ))

    print(
        "[INFO] Loaded SCP_static_drone_dyn replay: "
        f"{u_exec.shape[0]} controls @ dt={dt_plan:.3f}s, "
        f"command horizon={duration_cmd:.2f}s, total duration={total_duration:.2f}s, "
        f"replay_mode={replay_mode}, physics_mode={physics_mode}, scp_seed={seed_for_scp}"
    )

    pos_track_err = []
    vel_track_err = []
    rp_track_err_deg = []
    cmd_track_err = []

    # Command-tracking state for dynamic replay, matched to theory actuator lag model.
    cmd_state = x_exec[0, 6:9].copy()

    start_wall = time.time()
    i = 0
    while i < max_steps:
        if freeze_target:
            freeze_target_drone(env, target_pos)

        sim_t = i * env.CTRL_TIMESTEP
        x_ref = sample_state_linear(x_exec, sim_t, dt_plan)
        x_eval = x_ref

        if replay_mode == "pybullet_dynamic":
            chaser = obs[0]
            if sim_t < duration_cmd:
                u_target = sample_control_zoh(u_exec, sim_t, dt_plan)
            else:
                u_target = np.array([0.0, 0.0, scp["gravity"]])

            cmd_state = apply_theory_actuator_lag(
                cmd_state=cmd_state,
                cmd_target=u_target,
                dt_ctrl=env.CTRL_TIMESTEP,
                max_tilt_rad=scp["max_tilt"],
                u_min=scp["u_min"],
                u_max=scp["u_max"],
            )
            action[0] = nmpc_cmd_to_rpm(
                u_cmd=cmd_state,
                obs=chaser,
                env=env,
                max_tilt_rad=scp["max_tilt"],
                yaw_des_rad=DEFAULT_YAW_RAD,
            )
            action[1] = np.full(4, env.HOVER_RPM)
            obs, _, _, _, _ = env.step(action)
        else:
            # Kinematic replay: advance sim clock, then force chaser to theory state.
            action[0] = np.full(4, env.HOVER_RPM)
            action[1] = np.full(4, env.HOVER_RPM)
            obs, _, _, _, _ = env.step(action)
            x_ref_next = sample_state_linear(x_exec, sim_t + env.CTRL_TIMESTEP, dt_plan)
            set_chaser_from_reference(env, x_ref_next, yaw_des_rad=DEFAULT_YAW_RAD)
            x_eval = x_ref_next
            if freeze_target:
                freeze_target_drone(env, target_pos)
            obs = env._computeObs()
            if sim_t < duration_cmd:
                u_target = sample_control_zoh(u_exec, sim_t + env.CTRL_TIMESTEP, dt_plan)
            else:
                u_target = np.array([0.0, 0.0, scp["gravity"]])
            cmd_state = x_ref[6:9].copy()

        if freeze_target:
            freeze_target_drone(env, target_pos)

        chaser_pos = obs[0][0:3]
        chaser_vel = obs[0][10:13]
        chaser_rpy = np.array(p.getEulerFromQuaternion(obs[0][3:7]))
        update_visual_sphere(chaser_hull, chaser_pos, pyb_client)
        update_visual_sphere(target_hull, target_pos, pyb_client)

        pos_track_err.append(float(np.linalg.norm(chaser_pos - x_eval[0:3])))
        vel_track_err.append(float(np.linalg.norm(chaser_vel - x_eval[3:6])))
        rp_track_err_deg.append(float(np.linalg.norm(chaser_rpy[0:2] - x_eval[6:8]) * 180.0 / np.pi))
        if replay_mode == "pybullet_dynamic":
            cmd_track_err.append(float(np.linalg.norm(cmd_state - u_target)))

        if gui:
            env.render()
            sync(i, start_wall, env.CTRL_TIMESTEP)
        i += 1

    rel_pos = obs[0][0:3] - target_pos
    rel_vel = obs[0][10:13]
    print(f"[RESULT] Simulated time (s): {i * env.CTRL_TIMESTEP:.2f}")
    print(f"[RESULT] Final relative position (m): {rel_pos}")
    print(f"[RESULT] Final relative velocity (m/s): {rel_vel}")
    print(f"[RESULT] Final docking distance (m): {np.linalg.norm(rel_pos):.4f}")
    print(f"[RESULT] Mean position tracking error vs SCP replay (m): {np.mean(pos_track_err):.4f}")
    print(f"[RESULT] Mean velocity tracking error vs SCP replay (m/s): {np.mean(vel_track_err):.4f}")
    print(f"[RESULT] Mean roll/pitch tracking error vs SCP replay (deg): {np.mean(rp_track_err_deg):.3f}")
    if cmd_track_err:
        print(f"[RESULT] Mean command mismatch ||u_applied-u_hist|| (SI): {np.mean(cmd_track_err):.4f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Animate SCP_static_drone_dyn.py by replaying its online-updated command history "
            "u_hist=[roll_cmd, pitch_cmd, thrust_acc_cmd] without translational PD; "
            "only attitude control is used for command-to-RPM mapping. "
            "Use replay_mode=theory_kinematic for exact trajectory replay without PyBullet drift."
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
        help="Extra time after control horizon using hover thrust command",
    )
    parser.add_argument(
        "--replay_mode",
        default=DEFAULT_REPLAY_MODE,
        type=str,
        choices=["theory_kinematic", "pybullet_dynamic"],
        help=(
            "theory_kinematic: force chaser to theory states (no model mismatch drift). "
            "pybullet_dynamic: apply lag-matched commands through PyBullet physics."
        ),
    )
    parser.add_argument(
        "--physics_mode",
        default=DEFAULT_PHYSICS_MODE,
        type=str,
        choices=["pyb", "dyn"],
        help="Physics backend for replay: pyb (Bullet step) or dyn (explicit dynamics mode).",
    )
    parser.add_argument(
        "--scp_seed",
        default=0,
        type=int,
        help="Random seed used when executing SCP_static_drone_dyn.py (-1 for random every run).",
    )
    parser.add_argument(
        "--freeze_target",
        default=True,
        type=str2bool,
        help="Keep static target fixed at true position",
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
        help="Allow SCP_static_drone_dyn.py plots to display instead of suppressing them",
    )
    run(**vars(parser.parse_args()))
