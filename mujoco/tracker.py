import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the Scene
model = mujoco.MjModel.from_xml_path("scene.xml") 
data = mujoco.MjData(model)

MASS = 0.027
GRAVITY = 9.81
HOVER_FORCE = MASS * GRAVITY

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        
    def compute(self, pos_err, vel_err):
        # Uses true velocity error, preventing "Derivative Kick" on target changes
        return self.kp * pos_err + self.kd * vel_err

# =====================================================================
# TUNING PARAMETERS (Crucial for tiny 27g drone)
# =====================================================================
# Outer Loop (Position -> Target Acceleration)
# Outputs desired acceleration in m/s^2
pd_x = PDController(kp=3.0, kd=2.5)
pd_y = PDController(kp=3.0, kd=2.5)
pd_z = PDController(kp=8.0, kd=4.0) 

# Inner Loop (Angles -> Normalized Torques)
# Outputs raw control signals [-1, 1]
# Inner Loop (Angles -> Normalized Torques)
pd_roll  = PDController(kp=200.0, kd=50.0) 
pd_pitch = PDController(kp=200.0, kd=50.0)
pd_yaw   = PDController(kp=100.0, kd=25.0)

MAX_TILT = np.radians(25) # Hard limit tilt to 25 degrees to prevent flipping

# =====================================================================
# DUMMY TRAJECTORY
# =====================================================================
dt_sim = model.opt.timestep 
total_steps = 6000          

x_ref = np.zeros((total_steps, 3))
hover_steps = int(2.0 / dt_sim)
x_ref[:hover_steps] = [0.0, 0.0, 1.0]

move_steps = int(4.0 / dt_sim)
x_ref[hover_steps : hover_steps+move_steps, 0] = np.linspace(0.0, 2.0, move_steps)
x_ref[hover_steps : hover_steps+move_steps, 1] = np.linspace(0.0, 2.0, move_steps)
x_ref[hover_steps : hover_steps+move_steps, 2] = 1.0 

x_ref[hover_steps+move_steps:] = [2.0, 2.0, 1.0]

# =====================================================================
# SIMULATION LOOP
# =====================================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_idx = 0
    
    while viewer.is_running() and step_idx < total_steps:
        step_start = time.time()
        
        # --- 1. SENSOR READINGS ---
        pos = data.qpos[0:3]
        quat = data.qpos[3:7] 
        
        lin_vel = data.qvel[0:3] # True linear velocity
        ang_vel = data.qvel[3:6] # True angular velocity (p, q, r)
        
        w, xq, yq, zq = quat
        roll  = np.arctan2(2*(w*xq + yq*zq), 1 - 2*(xq**2 + yq**2))
        pitch = np.arcsin(2*(w*yq - zq*xq))
        yaw   = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq**2 + zq**2))
        

        
        
        target_pos = x_ref[step_idx]
        
        # Calculate the actual speed of the dummy trajectory
        if step_idx > 0:
            target_vel = (x_ref[step_idx] - x_ref[step_idx-1]) / dt_sim
        else:
            target_vel = np.zeros(3)
        
        # --- 3. OUTER LOOP (Position -> Acceleration -> Angles) ---
        err_pos = target_pos - pos
        err_vel = target_vel - lin_vel
        
        acc_des_x = pd_x.compute(err_pos[0], err_vel[0])
        acc_des_y = pd_y.compute(err_pos[1], err_vel[1])
        acc_des_z = pd_z.compute(err_pos[2], err_vel[2])
        
        # Convert accelerations to desired angles (Linearized hover assumption)
        # Pitching DOWN (negative) moves +X. Rolling RIGHT (positive) moves +Y.
        target_pitch = acc_des_x / GRAVITY
        target_roll  = -acc_des_y / GRAVITY
        target_yaw   = 0.0
        
        target_pitch = np.clip(target_pitch, -MAX_TILT, MAX_TILT)
        target_roll  = np.clip(target_roll, -MAX_TILT, MAX_TILT)

        # --- 4. INNER LOOP (Angles -> Torques) ---
        err_roll  = target_roll - roll
        err_pitch = target_pitch - pitch
        err_yaw   = target_yaw - yaw
        
        # Angular velocity targets are roughly 0 to damp motion
        err_droll  = 0.0 - ang_vel[0]
        err_dpitch = 0.0 - ang_vel[1]
        err_dyaw   = 0.0 - ang_vel[2]

        u_roll  = pd_roll.compute(err_roll, err_droll)
        u_pitch = pd_pitch.compute(err_pitch, err_dpitch)
        u_yaw   = pd_yaw.compute(err_yaw, err_dyaw)
        
        # --- 5. ACTUATOR MIXING ---
        total_thrust = MASS * (GRAVITY + acc_des_z)
        data.ctrl[0] = np.clip(total_thrust, 0, 0.35) 
        
        # XML gears are negative, so we invert the signal
        data.ctrl[1] = -u_roll
        data.ctrl[2] = -u_pitch
        data.ctrl[3] = -u_yaw

        # --- 6. DEBUG LOGGING (Print every ~0.5 seconds) ---
        if step_idx % 250 == 0:
            print(f"--- Step {step_idx} | Time {step_idx * dt_sim:.2f}s ---")
            print(f"Target Pos: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
            print(f"Actual Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"Actual Vel: [{lin_vel[0]:.2f}, {lin_vel[1]:.2f}, {lin_vel[2]:.2f}]")
            print(f"Target RPY: [R:{np.degrees(target_roll):.1f}°, P:{np.degrees(target_pitch):.1f}°, Y:{np.degrees(target_yaw):.1f}°]")
            print(f"Actual RPY: [R:{np.degrees(roll):.1f}°, P:{np.degrees(pitch):.1f}°, Y:{np.degrees(yaw):.1f}°]")
            print(f"Controls:   Thrust={data.ctrl[0]:.3f}, Roll={data.ctrl[1]:.3f}, Pitch={data.ctrl[2]:.3f}\n")

        mujoco.mj_step(model, data)
        viewer.sync()
        step_idx += 1
        
        # Real-time sync
        time_until_next_step = dt_sim - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)