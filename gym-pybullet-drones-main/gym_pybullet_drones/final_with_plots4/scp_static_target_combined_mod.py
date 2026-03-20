import time
import threading
import numpy as np
import cvxpy as cp
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ======================================================================
# 0. CONFIGURATION
# ======================================================================
SIM_FREQ = 240
CTRL_FREQ = 48
PLAN_INTERVAL = 0.2     
HORIZON = 20            
DT_PLAN = 0.1           
DURATION_SEC = 20.0     

DOCKING_AXIS = np.array([0.0, 0.0, -1.0]) 
CONE_ANGLE   = 30    
SAFETY_R     = 0.1   
ALPHA_LIMIT  = 1.05  

WIND_NOMINAL  = np.array([0.5, -0.3, -0.1]) 
WIND_GUST_AMP = 0.1                         

STATE_TRACKING    = 0
STATE_BACKING_OFF = 1
STATE_REPLANNING  = 2

# ======================================================================
# 1. STATIC TARGET
# ======================================================================
class StaticTarget:
    def get_state(self, t):
        s = np.zeros(6)
        s[0] = 0.5; s[1] = 0.0; s[2] = 1.0 
        return s

# ======================================================================
# 2. TARGET EKF (9D: Pos, Vel, Acc)
# ======================================================================
class TargetEKF:
    def __init__(self, dt):
        self.x = np.zeros(9)
        self.F = np.eye(9)
        self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.F[0,6]=0.5*dt**2; self.F[1,7]=0.5*dt**2; self.F[2,8]=0.5*dt**2
        self.F[3,6]=dt; self.F[4,7]=dt; self.F[5,8]=dt
        
        self.H = np.zeros((3,9)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(3) * 0.05
        self.P = np.eye(9) * 0.1

    def step(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.x

    def predict_future(self, steps, dt_plan):
        fut = np.zeros((steps, 9))
        tmp = self.x.copy()
        Fp = np.eye(9)
        Fp[0,3]=dt_plan; Fp[1,4]=dt_plan; Fp[2,5]=dt_plan
        Fp[0,6]=0.5*dt_plan**2; Fp[1,7]=0.5*dt_plan**2; Fp[2,8]=0.5*dt_plan**2
        Fp[3,6]=dt_plan; Fp[4,7]=dt_plan; Fp[5,8]=dt_plan
        for i in range(steps): 
            tmp = Fp @ tmp
            fut[i,:] = tmp
        return fut

# ======================================================================
# 3. SAFETY & VISUALS
# ======================================================================
def solve_dcol_scaling(p1, r1, p2, r2):
    return np.linalg.norm(p1 - p2) / (r1 + r2)

def create_hull(radius, color, client):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[0,0,0], physicsClientId=client)

def update_hull(body, pos, client):
    p.resetBasePositionAndOrientation(body, pos, [0,0,0,1], physicsClientId=client)
    
def draw_dynamic_cone(center, axis, angle_deg, client):
    if not hasattr(draw_dynamic_cone, "ids"): draw_dynamic_cone.ids = []
    axis = axis / np.linalg.norm(axis); cone_dir = -axis; length = 1.0; theta = np.deg2rad(angle_deg)
    ref = np.array([0,0,1]) if np.abs(axis[2]) < 0.9 else np.array([0,1,0])
    u = np.cross(axis, ref); u = u/np.linalg.norm(u); v = np.cross(axis, u)
    
    line_idx = 0
    for phi in np.linspace(0, 2*np.pi, 30):
        radial = u*np.cos(phi) + v*np.sin(phi)
        end = center + cone_dir * np.cos(theta)*length + radial * np.sin(theta)*length
        if len(draw_dynamic_cone.ids) <= line_idx:
            draw_dynamic_cone.ids.append(p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, physicsClientId=client))
        else:
            p.addUserDebugLine(center, end, [0,1,0], 2, lifeTime=0, replaceItemUniqueId=draw_dynamic_cone.ids[line_idx], physicsClientId=client)
        line_idx += 1

# ======================================================================
# 4. ASYNC PLANNER (FULLY ALIGNED WITH PAPER MATH)
# ======================================================================
class AsyncPlanner(threading.Thread):
    def __init__(self, N, dt, p_obs, r_obs, cone_angle, axis):
        super().__init__()
        self.N = N; self.dt = dt; self.p_obs = p_obs; self.r_obs = r_obs
        self.cone_angle = cone_angle; self.axis = axis
        self.daemon = True; self.lock = threading.Lock()
        self.req = None; self.res = None; self.prev_sol = None
        self.prev_u = None
        self.A = np.eye(6); self.A[0,3]=dt; self.A[1,4]=dt; self.A[2,5]=dt
        self.B = np.zeros((6,3)); self.B[0,0]=0.5*dt**2; self.B[1,1]=0.5*dt**2; self.B[2,2]=0.5*dt**2; self.B[3,0]=dt; self.B[4,1]=dt; self.B[5,2]=dt

    def request(self, chaser, preds, phase):
        with self.lock: self.req = (chaser, preds, phase)
        
    def get(self):
        with self.lock: return self.res

    def run(self):
        print("[Planner] Thread Started. Solver: SCS")
        while True:
            data = None
            with self.lock:
                if self.req: data = self.req; self.req = None
            if data:
                try:
                    traj_x, traj_u = self._solve_scp(data[0], data[1], data[2])
                    if traj_x is not None:
                        with self.lock: self.res = (traj_x, traj_u)
                except Exception as e: 
                    # If it fails, print the math error so it's not silent!
                    print(f"[SCP ERROR] {e}")
            time.sleep(0.01)

    def _solve_scp(self, start, preds, phase):
        if self.prev_sol is None:
            x_ref = np.zeros((6, self.N))
            for k in range(self.N):
                al = k/(self.N-1)
                x_ref[0:3,k] = (1-al)*start[0:3] + al*preds[-1,0:3]
        else:
            x_ref = np.zeros((6, self.N))
            x_ref[:, :-1] = self.prev_sol[:, 1:]
            x_ref[:, -1] = np.hstack([preds[-1, 0:3], preds[-1, 3:6]])

        cos_theta = np.cos(np.deg2rad(self.cone_angle))
        n_app = self.axis / np.linalg.norm(self.axis)

        # PAPER ALIGNMENT: Relaxed Q_mat to let the drone build velocity. 
        Q_mat = np.diag([1, 1, 5 , 1, 1, 1])  
        R_mat = np.diag([0.1, 0.1, 0.1])        
        V_MAX = 5.0                             
        U_MAX = 15.0  # Actuator Bound           
        Z_MIN = 0.1   # Altitude Bound
        
        # Target safety bubble allowed to reach 0 in terminal phase
        R_MIN_TARGET = 0.0 if phase == 1 else 0.25 

        u_ref = None
        for _ in range(2):
            x = cp.Variable((6, self.N))
            u = cp.Variable((3, self.N-1))
            
            # Slacks
            slack_obs = cp.Variable(self.N, nonneg=True)
            slack_cone = cp.Variable(self.N, nonneg=True)
            slack_target = cp.Variable(self.N, nonneg=True) 
            
            cost = 0
            con = [x[:,0] == start]
            
            # Terminal Relative Velocity Constraint
            con += [x[3:6, -1] == preds[-1, 3:6]] 
            
            for k in range(self.N-1):
                con += [x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k]]
                
                target_state = np.hstack([preds[k+1, 0:3], preds[k+1, 3:6]])
                cost += cp.quad_form(x[:, k+1] - target_state, Q_mat)
                cost += cp.quad_form(u[:,k], R_mat)

                con += [x[2, k+1] >= Z_MIN]                 
                con += [cp.norm(x[3:6, k+1]) <= V_MAX]      
                con += [cp.abs(u[:,k]) <= U_MAX]            

                p_ref = x_ref[0:3, k+1]; p_tar = preds[k+1, 0:3]
                vec_tar = p_ref - p_tar; dist_tar = np.linalg.norm(vec_tar) + 1e-4
                n_tar = vec_tar / dist_tar
                con += [n_tar @ (x[0:3, k+1] - p_tar) >= R_MIN_TARGET - slack_target[k+1]]

            for k in range(1, self.N):
                vec = x_ref[0:3,k] - self.p_obs
                n = vec / (np.linalg.norm(vec)+1e-4)
                con += [n @ (x[0:3,k] - self.p_obs) >= self.r_obs - slack_obs[k]]
                
                p_rel = x[0:3,k] - preds[k, 0:3] 
                dist_long = -n_app @ p_rel
                con += [dist_long >= 0]
                con += [cp.norm(p_rel) * cos_theta <= dist_long + slack_cone[k]]
                
                # Trust Region relaxed to prevent mathematically trapping the drone
                con += [cp.norm(x[:,k] - x_ref[:,k]) <= 5.0] 

            # Penalty scaled down to 1000 to prevent numerical explosion in the SCS solver
            cost += cp.sum(slack_obs)*1000 + cp.sum(slack_cone)*1000 + cp.sum(slack_target)*1000
            
            prob = cp.Problem(cp.Minimize(cost), con)
            try:
                prob.solve(solver=cp.SCS)
            except Exception as e:
                print(f"[Planner] SCS Solve failed: {e}")
            
            if x.value is not None:
                x_ref = x.value
                u_ref = u.value
            else:
                break

        if x_ref is not None and u_ref is not None:
            self.prev_sol = x_ref
            self.prev_u = u_ref
            return x_ref, u_ref
        else:
            return None, None

def run():
    SIM_FREQ = 240
    CTRL_FREQ = 48
    DURATION_SEC = 20.0
    
    CHASER_START = np.array([-2.5, 0.0, 1.5])
    P_OBS        = np.array([-1.0, 0.0, 1.25]) 
    R_OBS        = 0.4
    
    target_gen = StaticTarget()
    ekf = TargetEKF(dt=1/CTRL_FREQ)
    
    init_s = target_gen.get_state(0)
    init_9d = np.zeros(9); init_9d[0:6] = init_s 
    ekf.x = init_9d 
    
    planner = AsyncPlanner(HORIZON, DT_PLAN, P_OBS, R_OBS, CONE_ANGLE, DOCKING_AXIS)
    planner.start()
    
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=2, initial_xyzs=np.array([CHASER_START, init_s[0:3]]),
                     physics=Physics.PYB_DW, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ, gui=True, obstacles=False)
    PYB = env.getPyBulletClient()
    
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=R_OBS),
                      p.createVisualShape(p.GEOM_SPHERE, radius=R_OBS, rgbaColor=[1,0,0,0.2]), P_OBS, physicsClientId=PYB)
    hull_c = create_hull(SAFETY_R, [0, 1, 1, 0.3], PYB)
    hull_t = create_hull(SAFETY_R, [1, 0, 1, 0.3], PYB)
    
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    action = np.zeros((2,4))
    
    state = STATE_TRACKING
    curr_traj = None; curr_u = None
    last_plan = -1.0; backoff_start = None  # FIX 1: Triggers immediate plan on frame 1!
    docking_phase = 0 
    DOCK_OFFSET = np.array([-0.2, 0.0, 0.3]) 
    
    frozen = False
    
    history = {'t': [], 'p_c': [], 'p_t': [], 'v_c': [], 'action': []}
    START = time.time()
    
    try:
        for i in range(int(DURATION_SEC * CTRL_FREQ)):
            sim_t = i / CTRL_FREQ
            if not p.isConnected(physicsClientId=PYB): break
                    
            if frozen:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], freeze_pos_c, [0,0,0,1], physicsClientId=PYB)
                p.resetBasePositionAndOrientation(env.DRONE_IDS[1], freeze_pos_t, [0,0,0,1], physicsClientId=PYB)
                env.render()
                sync(i, START, env.CTRL_TIMESTEP)
                continue
                
            obs, _, _, _, _ = env.step(action)
            true_state = target_gen.get_state(sim_t)
            p.resetBasePositionAndOrientation(env.DRONE_IDS[1], true_state[0:3], [0,0,0,1], physicsClientId=PYB)
            
            p_chaser = obs[0][0:3]; p_target = obs[1][0:3]
            update_hull(hull_c, p_chaser, PYB); update_hull(hull_t, p_target, PYB)
            
            wind = WIND_NOMINAL + np.random.uniform(-WIND_GUST_AMP, WIND_GUST_AMP, 3)
            p.applyExternalForce(env.DRONE_IDS[0], -1, wind, p_chaser, p.WORLD_FRAME, PYB)
            draw_dynamic_cone(p_target, DOCKING_AXIS, CONE_ANGLE, PYB)

            ekf.step(true_state[0:3])
            alpha_obs = solve_dcol_scaling(p_chaser, SAFETY_R, P_OBS, R_OBS)
            
            if docking_phase == 0:
                xy_dist = np.linalg.norm(p_chaser[0:2] - p_target[0:2])
                z_dist = p_chaser[2] - p_target[2]
                if xy_dist < 0.3 and z_dist < 0.3:
                    docking_phase = 1
                    print(f"[PHASE] {sim_t:.2f}s | Switched to Terminal Phase (Planner executing dive!)")

            if docking_phase == 1 and np.linalg.norm(p_chaser - p_target) < 0.15:
                frozen = True
                freeze_pos_c = p_chaser; freeze_pos_t = p_target
                print(f"\n[COMPLETE] DOCKED SUCCESSFULLY at {sim_t:.2f}s")
                continue

            if state == STATE_TRACKING:
                if alpha_obs < ALPHA_LIMIT:
                    state = STATE_BACKING_OFF
                    backoff_start = p_chaser.copy()
                    vec = p_chaser - P_OBS; vec = vec / (np.linalg.norm(vec)+1e-6)
                    backoff_end = p_chaser + vec*0.5 + np.array([0,0,0.5])
                    backoff_t_start = time.time()
                else:
                    res = planner.get()
                    if res is not None: 
                        curr_traj, curr_u = res 
                        for j in range(HORIZON-1):
                            p.addUserDebugLine(curr_traj[0:3,j], curr_traj[0:3,j+1], [0,0,1], 3, physicsClientId=PYB)

                    if sim_t - last_plan > PLAN_INTERVAL:
                        preds = ekf.predict_future(HORIZON, DT_PLAN)
                        if docking_phase == 0: preds[:, 0:3] += DOCK_OFFSET
                        
                        chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                        planner.request(chaser_st, preds, docking_phase)
                        last_plan = sim_t

                    if curr_traj is not None and curr_u is not None:
                        idx = min(3, curr_traj.shape[1]-1)
                        pt = curr_traj[0:3, idx] 
                        vt = curr_traj[3:6, idx]
                        
                        action[0], _, _ = ctrl[0].computeControlFromState(
                            control_timestep=env.CTRL_TIMESTEP, 
                            state=obs[0], 
                            target_pos=pt, 
                            target_vel=vt
                            # FIX 3: Target_acc removed so it doesn't crash PyBullet.
                        )
                    else:
                        action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], CHASER_START)
            
            elif state == STATE_BACKING_OFF:
                elapsed = time.time() - backoff_t_start; progress = min(elapsed / 2.0, 1.0)
                k = progress * progress * (3 - 2 * progress)
                setpoint = (1-k)*backoff_start + k*backoff_end
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], setpoint)
                if progress >= 1.0: state = STATE_REPLANNING
                    
            elif state == STATE_REPLANNING:
                action[0], _, _ = ctrl[0].computeControlFromState(env.CTRL_TIMESTEP, obs[0], p_chaser)
                if sim_t - last_plan > 0.1:
                    preds = ekf.predict_future(HORIZON, DT_PLAN)
                    if docking_phase == 0: preds[:, 0:3] += DOCK_OFFSET
                    chaser_st = np.hstack([obs[0][0:3], obs[0][10:13]])
                    planner.request(chaser_st, preds, docking_phase)
                    last_plan = sim_t
                res = planner.get()
                if res is not None: state = STATE_TRACKING
            
            action[1] = np.zeros(4)
            history['t'].append(sim_t); history['p_c'].append(obs[0][0:3].copy()); history['p_t'].append(true_state[0:3].copy())
            env.render(); sync(i, START, env.CTRL_TIMESTEP)
            
    except Exception as e: print(f"\n[INFO] Simulation ended: {e}")
    finally:
        try: env.close()
        except: pass

if __name__ == "__main__":
    run()