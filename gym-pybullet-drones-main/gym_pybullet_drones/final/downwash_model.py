# ../examples/downwash.py
import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


def initialize_controllers_and_waypoints(
    drone_model: DroneModel,
    initial_positions: np.ndarray,
    waypoint_generator,
    control_freq_hz: int,
):
    """
    Generic initialization of controllers and waypoint data.

    Parameters
    ----------
    drone_model : DroneModel
        Drone model used by the controllers

    initial_positions : ndarray, shape (N, 3)
        Initial positions of N drones

    waypoint_generator : callable
        Function f(num_steps) -> ndarray (num_steps, d)
        Generates waypoint sequence

    control_freq_hz : int
        Control frequency in Hz

    Returns
    -------
    initial_positions : ndarray, shape (N, 3)

    waypoints : ndarray, shape (T, d)

    waypoint_indices : ndarray, shape (N,)

    controllers : list[DSLPIDControl]
    """

    num_drones = initial_positions.shape[0]

    # Generate waypoints (external logic)
    waypoints = waypoint_generator(control_freq_hz)

    # One counter per drone (no phase logic imposed)
    waypoint_indices = np.zeros(num_drones, dtype=int)

    # Controllers
    controllers = [
        DSLPIDControl(drone_model=drone_model)
        for _ in range(num_drones)
    ]

    return initial_positions, waypoints, waypoint_indices, controllers




# Demo use case
# def circular_waypoints(control_freq_hz):
#     period = 5.0
#     num_wp = int(period * control_freq_hz)
#     wps = np.zeros((num_wp, 2))
#     for k in range(num_wp):
#         wps[k] = [0.5 * np.cos(2*np.pi*k/num_wp), 0.0]
#     return wps
# init_xyzs = np.array([
#     [ 0.5, 0.0, 1.0],
#     [-0.5, 0.0, 0.5]
# ])

# init_xyzs, waypoints, wp_idx, ctrls = initialize_controllers_and_waypoints(
#     drone_model=DroneModel.CF2X,
#     initial_positions=init_xyzs,
#     waypoint_generator=circular_waypoints,
#     control_freq_hz=48
# )

