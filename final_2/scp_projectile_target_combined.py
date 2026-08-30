import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================================
# 0. CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     # Fast planning (5Hz)
HORIZON = 20            # 2.0s Horizon
DT_PLAN = 0.1           
DURATION_SEC = 20.0     

# --- DOCKING & SAFETY ---
DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  

# Wind
WIND_NOMINAL  = np.array([0.5, -0.3, -0.1]) 
WIND_GUST_AMP = 0.1                         

# FSM States
STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. PROJECTILE TARGET (More Horizontal, Less Vertical)
# ======================================================================
class ProjectileTarget:
    def get_state(self, t):
        s = np.zeros(6)
        
        # Initial Conditions
        x_0 = 0.5
        z_0 = 2.0
        
        # MODIFICATIONS:
        # Move slowly: vx is low (0.1 to 0.2)
        # Range >> Depth: g must be extremely small relative to vx
        vx  = 0.15   # Reduced from 0.5 for "very slow" motion
        g   = 0.005  # Reduced from 0.05 to make it "super floaty" 
        
        # Projectile Physics
        s[0] = x_0 + vx * t          # X: Linear increase (Slow)
        s[1] = 0.0                   
        s[2] = z_0 - 0.5 * g * t**2  # Z: Parabolic drop (Extremely slow)
        
        # Floor clamp
        if s[2] < 0.05:
            s[2] = 0.05
            s[5] = 0.0
        else:
            s[5] = -g * t  
            
        s[3] = vx      
        s[4] = 0.0     
        
        return s

# ======================================================================
# 2. TARGET EKF
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(6)
        self.F = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(6)*0.01; self.R = np.eye(3)*0.05
        self.P = np.eye(6)*0.1

    def step(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 6)); tmp = self.x.copy()
        Fp = np.eye(6); Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        for i in range(steps): tmp = Fp @ tmp; fut[i,:] = tmp
        return fut

# ======================================================================
# 3. SAFETY & VISUALS
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    dist = np.linalg.norm(p1 - p2)
    return dist / (r1 + r2)

def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)
    return body

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)
    
    
def draw_dynamic_cone(center, axis, angle_deg, client):
    # Initialize static storage for line IDs if it doesn't exist
    if not hasattr(draw_dynamic_cone, "ids"):
        draw_dynamic_cone.ids = []

    # Axis [0,0,-1] -> Approach [0,0,1]
    axis = axis / np.linalg.norm(axis)
    cone_dir = -axis 
    
    length = 1.0 # Short cone to visualize immediate docking area
    theta = np.deg2rad(angle_deg)
    
    if np.abs(axis[2]) < 0.9: ref = np.array([0,0,1])
    else: ref = np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u)
    v = np.cross(axis, u)
    
    line_idx = 0
    # Draw Radial Lines (30 lines)
    for phi in np.linspace(0, 2*np.pi, 30):
        radial = u*np.cos(phi) + v*np.sin(phi)
        vec = cone_dir * np.cos(theta) + radial * np.sin(theta)
        end = center + vec * length
        
        if len(draw_dynamic_cone.ids) <= line_idx:
            # Create new line (lifeTime=0 makes it persistent)
            uid = p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, physicsClientId=client)
            draw_dynamic_cone.ids.append(uid)
        else:
            # Update existing line
            p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
        
        line_idx += 1

    # Draw Center Axis Line
    if len(draw_dynamic_cone.ids) <= line_idx:
        uid = p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, physicsClientId=client)
        draw_dynamic_cone.ids.append(uid)
    else:
        p.addUserDebugLine(center, center + cone_dir*length, [1,0,0], 5, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
    

# ======================================================================
# 4. ASYNC PLANNER
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt

    def request(self, chaser, preds):
        with self.lock: self.req = (chaser, preds)
    def get(self):
        with self.lock: return self.res

    def run(self):
        print("[Planner] Thread Started.")
        while True:
            data = None
            with self.lock:
                if self.req: data = self.req; self.req = None
            if data:
                try:
                    traj = self._solve_scp(data[0], data[1])
                    with self.lock: self.res = traj
                except: pass
            time.sleep(0.01)

    def _solve_scp(self, start, preds):
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*start[0:3] + al*preds[-1,0:3]
                x_ref[1,k] += 1.0 * np.sin(np.pi*al) # Arc for start
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = preds[-1]

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            slack_obs = cp.Variable(self.N, nonneg=True)
            slack_cone = cp.Variable(self.N, nonneg=True)
            
            # AIMING FOR EXACT DOCKING (0.0 offset)
            # We want to hit the target itself eventually
            dock_offset = np.zeros(3) 
            
            cost = 0
            con = [x[:,0] == start]
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                # --- COST TUNING FOR DOCKING ---
                # 1. Strict Position Tracking
                target_k = preds[k+1, 0:3] + dock_offset
                cost += 50 * cp.sum_squares(x[0:3, k+1] - target_k)
                
                # 2. Minimal Control (but not too cheap, we need agility)
                cost += 0.1 * cp.sum_squares(u[:,k])

            for k in range(1, self.N):
                # Obstacle
                vec = x_ref[0:3,k] - self.p_obs
                n = vec / (np.linalg.norm(vec)+1e-4)
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack_obs[k]]
                
                # Cone Constraint
                p_rel = x[0:3,k] - preds[k, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]
                
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.5]

            cost += cp.sum(slack_obs)*1e6 + cp.sum(slack_cone)*1e5
            prob = cp.Problem(cp.Minimize(cost), con)
            try: prob.solve(solver=cp.OSQP); 
            except: prob.solve(solver=cp.SCS)
            
            if x.value is None: break
            x_ref = x.value

        self.prev_sol = x_ref
        return x_ref


def plot_performance(history):
    t = np.array(history['t'])
    p_c = np.array(history['p_c'])
    p_t = np.array(history['p_t'])
    v_c = np.array(history['v_c'])
    act = np.array(history['action'])
    
    # Calculate tracking error norm over time
    err = np.linalg.norm(p_c - p_t, axis=1)

    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(p_t[:,0], p_t[:,1], p_t[:,2], 'g--', label='Target')
    ax1.plot(p_c[:,0], p_c[:,1], p_c[:,2], 'b-', linewidth=2, label='Chaser')
    ax1.scatter(p_c[0,0], p_c[0,1], p_c[0,2], c='k', marker='o', label='Start')
    ax1.scatter(p_c[-1,0], p_c[-1,1], p_c[-1,2], c='r', marker='*', s=100, label='End')
    ax1.set_title("3D Trajectory")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend()

    # 2. Position X, Y, Z vs Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, p_t[:,0], 'g--', alpha=0.5)
    ax2.plot(t, p_c[:,0], 'b-', label='X')
    ax2.plot(t, p_t[:,1], 'g--', alpha=0.5)
    ax2.plot(t, p_c[:,1], 'r-', label='Y')
    ax2.plot(t, p_t[:,2], 'g--', alpha=0.5)
    ax2.plot(t, p_c[:,2], 'k-', label='Z')
    ax2.set_title("Position Evolution (Target dashed)")
    ax2.grid(True)
    ax2.legend()

    # 3. Docking Error (Distance)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, err, 'r-', linewidth=2)
    ax3.axhline(0.10, color='k', linestyle=':', label='Dock Threshold')
    ax3.set_title("Euclidean Distance to Target")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Distance (m)")
    ax3.grid(True)
    ax3.legend()

    # 4. Velocity
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, v_c[:,0], label='Vx')
    ax4.plot(t, v_c[:,1], label='Vy')
    ax4.plot(t, v_c[:,2], label='Vz')
    ax4.set_title("Chaser Velocities")
    ax4.grid(True)
    ax4.legend()

    # 5. Control Actions (RPM/Thrust)
    ax5 = fig.add_subplot(2, 3, (5, 6))
    ax5.plot(t, act[:,0], label='M1')
    ax5.plot(t, act[:,1], label='M2')
    ax5.plot(t, act[:,2], label='M3')
    ax5.plot(t, act[:,3], label='M4')
    ax5.set_title("Control Inputs (RPM/Thrust)")
    ax5.set_xlabel("Time (s)")
    ax5.grid(True)
    ax5.legend(loc='upper right', ncol=4)

    plt.tight_layout()
    plt.show()
    
    
    
# ======================================================================
# 5. MAIN EXECUTION
# ======================================================================
def run():
    CHASER_START = np.array([-2.5, 0.0, 2.0])
    OBS_POS = np.array([-0.5, 0.0, 1.0]); OBS_RAD = 0.4
    
    target_gen = ProjectileTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    init_s = target_gen.get_state(0)
    ekf.x = init_s # Init perfect
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, OBS_POS, OBS_RAD, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, 
        num_drones=2,
        initial_xyzs=np.array([CHASER_START, init_s[0:3]]),
        physics=Physics.PYB_DW, # Downwash
        pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=True, obstacles=False
    )
    PYB = env.getPyBulletClient()
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RAD),
                      p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RAD, rgbaColor=[1,0,0,0.9]),
                      OBS_POS, physicsClientId=PYB)
    
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    state = STATE_TRACKING
    curr_traj = None
    last_plan = 0
    backoff_start, backoff_end, backoff_t_start = None, None, 0
    
    # --- PHASE LOGIC ---
    
    docking_phase = 0 # 0: Approach Offset, 1: Terminal Docking
    DOCK_OFFSET = np.array([0, 0.0, 0.3]) # 2m up in cone
    frozen = False
    freeze_pos_c = None
    freeze_pos_t = None
    
    print("[SIM] Running Two-Phase Docking (Offset -> Terminal)...")
    print("      Target Type: Projectile (Horizontal)")
    START = time.time()
    
    for t in np.arange(0, DURATION_SEC, 0.1):
        p.addUserDebugLine(target_gen.get_state(t)[0:3], target_gen.get_state(t+0.1)[0:3], [0.5, 0, 0.5], 2, physicsClientId=PYB)
        
    
    # ADD THIS BLOCK
    history = {
        't': [], 
        'p_c': [], 'p_t': [], 
        'v_c': [], 'action': []
    }
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        sim_t = i / CTRL_FREQ
        
        # 0. FREEZE CHECK (If docked, we stop updates)
        if frozen:
            # Override visuals/physics to hold position
            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], freeze_pos_c, [0,0,0,1], physicsClientId=PYB)
            p.resetBasePositionAndOrientation(env.DRONE_IDS[1], freeze_pos_t, [0,0,0,1], physicsClientId=PYB)
            env.render()
            sync(i, START, env.CTRL_TIMESTEP)
            continue
        
        # 1. Physics
        obs, _, _, _, _ = env.step(action)
        true_state = target_gen.get_state(sim_t)
        
        # ADD THIS BLOCK
        history['t'].append(sim_t)
        history['p_c'].append(obs[0][0:3])
        history['p_t'].append(true_state[0:3])
        history['v_c'].append(obs[0][10:13])
        history['action'].append(action[0])
        
        
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
        
        # Visuals
        p_chaser = obs[0][0:3]
        p_target = obs[1][0:3]
        update_hull(hull_c, p_chaser, PYB)
        update_hull(hull_t, p_target, PYB)
        
        wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
        p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
        p.addUserDebugLine(p_chaser, p_chaser + wind*0.5, [1,1,0], 2, lifeTime=0.1, physicsClientId=PYB)
        draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)
        
        # Visualize Offset Point
        # if docking_phase == 0:
            # p.addUserDebugLine(p_target, p_target + DOCK_OFFSET, [1,1,1], 1, lifeTime=0.1, physicsClientId=PYB)
            # p.addUserDebugText("Offset", p_target + DOCK_OFFSET + [0,0,0.1], [1,1,1], lifeTime=0.1, physicsClientId=PYB)

        # 2. EKF
        ekf.step(true_state[0:3] + np.random.normal(0, 0.01, 3))
        
        # 3. SAFETY
        alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, OBS_POS, OBS_RAD)
        
        # --- PHASE TRANSITION CHECK ---
        if docking_phase == 0:
            # Distance to the offset point
            dist_to_offset = np.linalg.norm(p_chaser - (p_target + DOCK_OFFSET))
            if dist_to_offset < 0.2:
                docking_phase = 1
                print(f"[PHASE] {sim_t:.2f}s | Switched to Terminal Phase (Phase 1)")

        # --- DOCKING COMPLETION CHECK ---
        dist_real = np.linalg.norm(p_chaser - p_target)
        if docking_phase == 1 and dist_real < 0.2:
            plot_performance(history)
            
            frozen = True
            freeze_pos_c = p_chaser
            freeze_pos_t = p_target
            print(f"[COMPLETE] {sim_t:.2f}s | DOCKED! FREEZING DRONES.")
            continue

        # 4. CONTROL LOGIC
        if state == STATE_TRACKING:
            
            # Backoff Condition
            if alpha_obs < ALPHA_LIMIT:
                print(f"[ALERT] Obstacle Collision Risk! Backing Off.")
                state = STATE_BACKING_OFF
                backoff_start = p_chaser.copy()
                vec = p_chaser - OBS_POS
                vec = vec / (np.linalg.norm(vec)+1e-6)
                backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                backoff_t_start = time.time()
                p.changeVisualShape(hull_c, -1, rgbaColor=[1, 0, 0, 0.8], physicsClientId=PYB)
            
            else:
                # Plan
                res = planner.get()
                if res is not None: 
                    curr_traj = res
                    # Visuals for Trajectory
                    for j in range(HORIZON-1):
                        p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                if sim_t - last_plan > PLAN_INTERVAL:
                    preds = ekf.predict_future(HORIZON, DT_PLAN)
                    
                    # --- MODIFY TARGET FOR PLANNER BASED ON PHASE ---
                    if docking_phase == 0:
                        # In phase 0, we tell planner the target is at (True Target + Offset)
                        preds[:, 0:3] += DOCK_OFFSET
                    # In phase 1, we send raw preds (Target Center)
                    
                    chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                    planner.request(chaser_st, preds)
                    last_plan = sim_t
                    
                    # Draw Planner Goal
                    # for j in range(HORIZON-1):
                        # p.addUserDebugLine(preds[j,0:3], preds[j+1,0:3], [1,0,0], 1, physicsClientId=PYB)

                # --- TRACKING ---
                if curr_traj is not None:
                    idx = min(3, curr_traj.shape[1]-1)
                    pt = curr_traj[0:3, idx] 
                    vt = curr_traj[3:6, idx]
                    action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], pt, vt)
                else:
                    action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START, np.zeros(3))

        elif state == STATE_BACKING_OFF:
            elapsed = time.time() - backoff_t_start
            progress = min(elapsed / 2.0, 1.0)
            k = progress * progress * (3 - 2 * progress)
            setpoint = (1-k)*backoff_start + k*backoff_end
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], setpoint, np.zeros(3))
            
            if progress >= 1.0:
                state = STATE_REPLANNING
                
        elif state == STATE_REPLANNING:
            action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser, np.zeros(3))
            if sim_t - last_plan > 0.1:
                preds = ekf.predict_future(HORIZON, DT_PLAN)
                if docking_phase == 0: preds[:, 0:3] += DOCK_OFFSET
                chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                planner.request(chaser_st, preds)
                last_plan = sim_t
            
            res = planner.get()
            if res is not None:
                if np.linalg.norm(res[0:3,0] - p_chaser) < 0.3:
                    curr_traj = res
                    p.changeVisualShape(hull_c, -1, rgbaColor=[0, 1, 1, 0.3], physicsClientId=PYB)
                    state = STATE_TRACKING
        
        action[1] = np.zeros(4)
        
        if i % 20 == 0:
            c_pos = obs[0][0:3]
            t_pos = obs[1][0:3]
            dist = np.linalg.norm(c_pos - t_pos)
            p_str = "OFFSET" if docking_phase == 0 else "TERMINAL"
            print(f"{sim_t:05.2f} | Phase: {p_str} | Dist: {dist:.2f}m")

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    
    plot_performance(history) # Add here too for timeout cases
    
    env.close()

if __name__ == "__main__":
    run()