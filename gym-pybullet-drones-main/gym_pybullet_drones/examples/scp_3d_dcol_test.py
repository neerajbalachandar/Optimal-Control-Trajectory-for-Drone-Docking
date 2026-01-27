import time
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ======================================================================
# CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
SAFETY_R = 0.3  # Large radius to make collision obvious
ALPHA_LIMIT = 1.05 # Trigger if objects are within 5% of touching

# ======================================================================
# 1. DCOL SOLVER (Mathematical Collision Check)
# ======================================================================
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
        return 0.0, p1 # Fallback

# ======================================================================
# 2. VISUALIZATION
# ======================================================================
def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)

# ======================================================================
# 3. MAIN
# ======================================================================
def run():
    # Setup
    CHASER_START = np.array([-1.0, 0.0, 1.0])
    TARGET_POS   = np.array([ 1.0, 0.0, 1.0]) # Static Target
    
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=2,
                     initial_xyzs=np.array([CHASER_START, TARGET_POS]),
                     physics=Physics.PYB, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
                     gui=True)
    PYB = env.getPyBulletClient()
    
    # VISUALS
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.5], PYB) # Cyan
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.5], PYB) # Magenta
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    # Initial "Naive" Plan: Fly straight to target
    # 5 seconds to cross 2 meters
    steps = int(5.0 * CTRL_FREQ)
    waypoints = np.linspace(CHASER_START, TARGET_POS, steps)
    
    safe_mode = False
    stop_pos = None
    
    print("STARTING: Chaser moving to collision...")
    START = time.time()
    
    for i in range(steps * 2): # Run longer to see holding
        # 1. Physics
        obs, _, _, _, _ = env.step(action)
        pos_c = obs[0][0:3]
        pos_t = obs[1][0:3]
        
        # 2. Update Visuals
        update_hull(hull_c, pos_c, PYB)
        update_hull(hull_t, pos_t, PYB)
        
        # 3. DCOL CHECK
        alpha, contact_pt = solve_dcol_scaling(pos_c, SAFETY_R, pos_t, SAFETY_R)
        
        # 4. LOGIC TRIGGER
        if not safe_mode:
            if alpha < ALPHA_LIMIT:
                print(f"[ALERT] Alpha {alpha:.3f} < {ALPHA_LIMIT}. HULLS TOUCHING!")
                print(">>> SWITCHING TO SAFETY HOVER <<<")
                safe_mode = True
                stop_pos = pos_c.copy() # Capture current position to hold
                
                # Visual Marker of intersection
                p.addUserDebugLine(pos_c, pos_t, [1,0,0], 3, physicsClientId=PYB)
                p.addUserDebugText("COLLISION!", contact_pt, [1,0,0], textSize=2, physicsClientId=PYB)
                # Turn hull Red
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)

        # 5. CONTROL
        if safe_mode:
            # STOP AND REPLAN (Here: Just Hover)
            target_pt = stop_pos
        else:
            # Continue naive path
            idx = min(i, len(waypoints)-1)
            target_pt = waypoints[idx]

        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], target_pt, np.zeros(3))
        # Target stays still
        action[1], _, _ = ctrl[1].computeControlFromState(env.CTRL_TIMESTEP, obs[1], TARGET_POS, np.zeros(3))
        
        # Debug Print
        if i % 10 == 0:
            status = "SAFE" if not safe_mode else "COLLIDED"
            print(f"T={i/CTRL_FREQ:.2f} | Dist: {np.linalg.norm(pos_c - pos_t):.3f} | Alpha: {alpha:.3f} | {status}")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()

if __name__ == "__main__":
    run()