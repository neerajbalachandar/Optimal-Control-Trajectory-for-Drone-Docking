import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
N = 50               # Horizon steps
dt = 0.1             # Time step
p_start = np.array([0.0, 0.0])

# OBSTACLE (Static for this test)
obs_pos = np.array([2.5, -0.5])
obs_rad = 0.8

# --- 2. GENERATE MOVING TARGET TRAJECTORY ---
# The target is "Non-Cooperative" - it follows its own sine wave path.
# We assume we know this path (via prediction/sensing).
target_traj = np.zeros((4, N))
t_vec = np.arange(N) * dt

for k in range(N):
    # X: Linear motion 2.0 -> 5.0
    x_t = 2.0 + 3.0 * (k / (N - 1))
    # Y: Sine wave
    y_t = -1.0 + 1.0 * np.sin(2 * np.pi * (k / (N - 1)))
    
    target_traj[0:2, k] = [x_t, y_t]

# Calculate Target Velocities (Finite Differences)
# This is crucial so we can match speed for a "Soft Docking"
for k in range(N-1):
    target_traj[2:4, k] = (target_traj[0:2, k+1] - target_traj[0:2, k]) / dt
# Repeat last velocity for the final step
target_traj[2:4, N-1] = target_traj[2:4, N-2]

# --- 3. DYNAMICS (Double Integrator) ---
A = np.eye(4)
A[0, 2] = dt; A[1, 3] = dt
B = np.zeros((4, 2))
B[0, 0] = 0.5 * dt**2; B[1, 1] = 0.5 * dt**2
B[2, 0] = dt; B[3, 1] = dt

# --- 4. INITIAL GUESS (Intercept Heuristic) ---
# We guess a straight line from Start -> TARGET FINAL POSITION.
# If we aimed at the target's current position, we'd be "chasing our tail".
x_ref = np.zeros((4, N))
final_target_pos = target_traj[0:2, N-1]

for k in range(N):
    alpha = k / (N - 1)
    # Linear interpolation to the INTERCEPT point
    x_ref[0:2, k] = (1 - alpha) * p_start + alpha * final_target_pos

# --- 5. SCP LOOP ---
max_iters = 15
plt.figure(figsize=(10, 6))
print(f"{'Iter':<5} | {'Diff':<10} | {'Max Slack':<10}")
print("-" * 35)

for iteration in range(max_iters):
    # Variables
    x = cp.Variable((4, N))
    u = cp.Variable((2, N-1))
    slack = cp.Variable(N, nonneg=True)
    
    # Cost: Fuel + Heavy Obstacle Penalty
    cost = cp.sum(cp.norm(u, 1, axis=0)) + 100000 * cp.sum(slack)
    
    constraints = []
    
    # Dynamics
    constraints += [x[:, 0] == np.hstack([p_start, 0, 0])]
    for k in range(N-1):
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
        
    # --- TERMINAL CONSTRAINT (THE INTERCEPT) ---
    # We must match the Moving Target's exact state at step N
    # Position match = Collision/Docking
    # Velocity match = Soft Docking (Relative velocity = 0)
    constraints += [x[:, N-1] == target_traj[:, N-1]]
    
    # --- OBSTACLE AVOIDANCE ---
    for k in range(1, N-1):
        p_ref = x_ref[0:2, k]
        vec = p_ref - obs_pos
        dist = np.linalg.norm(vec)
        
        if dist < 0.1: 
            n_vec = np.array([0., 1.]) # Singularity fix
        else:
            n_vec = vec / dist
            
        # Linearized Constraint
        constraints += [n_vec @ (x[0:2, k] - obs_pos) >= obs_rad - slack[k]]
    
    # Trust Region
    for k in range(N):
        constraints += [cp.norm(x[:, k] - x_ref[:, k]) <= 2.0]
        
    # Input Limits
    for k in range(N-1):
        constraints += [cp.norm(u[:, k]) <= 5.0]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    
    if x.value is None:
        print("Solver Failed")
        break
        
    # Metrics
    diff = np.linalg.norm(x.value - x_ref)
    max_slack = np.max(slack.value)
    print(f"{iteration+1:<5} | {diff:.4f}     | {max_slack:.4f}")
    
    x_ref = x.value.copy()
    
    # --- PLOTTING ---
    plt.clf()
    
    # 1. Draw Target Path (Red Dashed)
    plt.plot(target_traj[0, :], target_traj[1, :], 'r--', linewidth=1.5, label='Target Path')
    # Draw Target at Final Intercept Point
    plt.plot(target_traj[0, -1], target_traj[1, -1], 'rh', markersize=12, label='Intercept Point')
    
    # 2. Draw Obstacle
    circle = plt.Circle(obs_pos, obs_rad, color='gray', alpha=0.5, label='Obstacle')
    plt.gca().add_patch(circle)
    
    # 3. Draw Chaser Path (Blue)
    plt.plot(x_ref[0, :], x_ref[1, :], 'b.-', linewidth=2, label='Chaser Path')
    plt.plot(p_start[0], p_start[1], 'go', markersize=10, label='Chaser Start')
    
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    plt.title(f"Iteration {iteration+1} | Moving Target Intercept")
    plt.pause(0.1)
    
    if diff < 0.1 and max_slack < 0.01:
        print("Converged! Trajectory Intercepts Target.")
        break

plt.show()