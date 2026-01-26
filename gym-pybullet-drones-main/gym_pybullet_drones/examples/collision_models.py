import numpy as np

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])

def signed_distance_oriented_box(p, c, q, half_extents):
    """
    Signed distance from point p to oriented box centered at c
    """
    R = quat_to_rot(q)
    local = R.T @ (p - c)
    d = np.abs(local) - half_extents
    outside = np.maximum(d, 0)
    inside = np.minimum(np.max(d), 0)
    return np.linalg.norm(outside) + inside
