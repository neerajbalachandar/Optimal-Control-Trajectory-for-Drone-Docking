import time
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

def solve_dcol_scaling(p1, r1, p2, r2):
    """
    Solves for the minimum scaling factor 'alpha' such that two spheres touch.
    If alpha < 1.0, they are colliding.
    """
    # For two spheres, this has a simple analytical solution:
    # alpha = distance / (r1 + r2)
    # But we use CVXPY to prove the optimization method works (as per paper)
    
    alpha = cp.Variable(nonneg=True)
    x = cp.Variable(3)
    
    constraints = [
        cp.norm(x - p1) <= alpha * r1,
        cp.norm(x - p2) <= alpha * r2
    ]
    
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    try:
        prob.solve(solver=cp.ECOS)
        return alpha.value, x.value
    except:
        return 0.0, p1
    

# Visualization of hull
def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)