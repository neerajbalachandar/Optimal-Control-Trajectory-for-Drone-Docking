import numpy as np

def get_discrete_dynamics(dt: float):
    """
    Returns discrete-time state-space matrices (A, B) for a
    3D double-integrator model with state:

        x = [px, py, pz, vx, vy, vz]^T
        u = [ax, ay, az]^T

    Dynamics:
        x_{k+1} = A x_k + B u_k

    Parameters
    ----------
    dt : float
        Discrete time step

    Returns
    -------
    A : ndarray, shape (6, 6)
        State transition matrix
    B : ndarray, shape (6, 3)
        Control input matrix
    """

    A = np.eye(6)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt

    B = np.zeros((6, 3))
    B[0, 0] = 0.5 * dt**2
    B[1, 1] = 0.5 * dt**2
    B[2, 2] = 0.5 * dt**2
    B[3, 0] = dt
    B[4, 1] = dt
    B[5, 2] = dt

    return A, B
