import numpy as np

# Crazyflie 2X parameters (consistent with assets/cf2x.urdf)
M = 0.027
G = 9.8
L_ARM = 0.0397
KF = 3.16e-10
KM = 7.94e-12
THRUST2WEIGHT_RATIO = 2.25

MAX_TILT_DEG = 35.0
MAX_TILT_RAD = np.radians(MAX_TILT_DEG)

F_HOVER = M * G
F_MAX = THRUST2WEIGHT_RATIO * F_HOVER
MAX_TOTAL_ACCEL = F_MAX / M

# Keep the original paper-style box bound as an optional additional limit.
DEFAULT_U_INF_MAX = 15.0


def build_double_integrator_matrices(dt: float):
    """Discrete-time translational dynamics for state [p, v].

    The translational kinematics are still double-integrator, but control inputs
    are constrained by physically-feasible drone acceleration limits elsewhere.
    """
    a_d = np.eye(6)
    a_d[0:3, 3:6] = dt * np.eye(3)

    b_d = np.zeros((6, 3))
    b_d[0:3, :] = 0.5 * dt**2 * np.eye(3)
    b_d[3:6, :] = dt * np.eye(3)
    return a_d, b_d


def drone_accel_constraints(cp, u_expr, u_inf_max: float = DEFAULT_U_INF_MAX):
    """Return convex constraints enforcing drone-feasible inertial acceleration.

    Let u be inertial acceleration (m/s^2). For a quadrotor,
    f/m = u + g*e3 must be realizable by bounded thrust and tilt.
    """
    e3 = np.array([0.0, 0.0, 1.0])
    specific_force = u_expr + G * e3

    con = [
        # Optional legacy control-box limit from prior SCP setup.
        cp.norm(u_expr, np.inf) <= u_inf_max,
        # Total specific force bounded by max thrust.
        cp.norm(specific_force, 2) <= MAX_TOTAL_ACCEL,
        # Positive body-z specific force so SOC RHS stays nonnegative.
        u_expr[2] + G >= 1e-3,
        # Tilt-feasible lateral acceleration envelope.
        cp.norm(u_expr[0:2], 2) <= np.tan(MAX_TILT_RAD) * (u_expr[2] + G),
    ]
    return con
