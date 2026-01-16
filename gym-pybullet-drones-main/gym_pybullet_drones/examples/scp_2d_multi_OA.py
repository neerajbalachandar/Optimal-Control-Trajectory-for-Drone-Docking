import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- 1. SETUP ---
N = 60                  # More steps for weaving
dt = 0.2
p_start = np.array([-5.0, 0.0])
p_goal = np.array([2.0, 3.0])

# Define Multiple Obstacles [x, y, radius]
# 1. Top-Left, 2. Bottom-Center, 3. Top-Right
obstacles = np.array([
    [-2.0,  1.5, 1.5],
    [ 0.0, -1.5, 1.5],
    [ 2.0,  1.5, 1.5]
])
n_obs = len(obstacles)

# Dynamics (Double Integrator)
A = np.eye(4)
A[0, 2] = dt; A[1, 3] = dt
B = np.zeros((4, 2))
B[0, 0] = 0.5 * dt**2; B[1, 1] = 0.5 * dt**2
B[2, 0] = dt; B[3, 1] = dt

# --- 2. INITIAL GUESS (Naive Straight Line) ---
# As per your MATLAB comments, we use a straight line to stress-test the SCP.
x_ref = np.zeros((4, N))
for k in range(N):
    alpha = k / (N - 1)
    x_ref[0:2, k] = (1 - alpha) * p_start + alpha * p_goal

# --- 3. SCP LOOP ---
max_iters = 20
plt.figure(figsize=(10, 6))

print(f"{'Iter':<5} | {'Step Diff':<10} | {'Total Slack':<10}")
print("-" * 35)

for iteration in range(max_iters):
    # Variables
    x = cp.Variable((4, N))
    u = cp.Variable((2, N-1))
    # Matrix of slack variables: One for each obstacle, at each timestep
    slack = cp.Variable((n_obs, N), nonneg=True)

    # Cost: Minimize Control + Heavy Obstacle Penalty
    cost = cp.sum_squares(u) + 100000 * cp.sum(slack)
    
    constraints = []
    
    # Dynamics Constraints
    constraints += [x[:, 0] == np.hstack([p_start, 0, 0])] # Start
    for k in range(N-1):
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
    
    # Terminal Constraints
    constraints += [cp.norm(x[0:2, N-1] - p_goal) <= 0.1]
    constraints += [x[2:4, N-1] == 0] # Stop at goal

    # --- MULTI-OBSTACLE AVOIDANCE ---
    # We iterate k from 1 to N-1 (Python 1:N)
    for k in range(1, N): 
        curr_pos_ref = x_ref[0:2, k]
        
        for obs_idx in range(n_obs):
            # Get specific obstacle data
            o_pos = obstacles[obs_idx, 0:2]
            o_rad = obstacles[obs_idx, 2]
            
            # Linearize around current reference
            vec = curr_pos_ref - o_pos
            dist = np.linalg.norm(vec)
            
            # Handle singularity
            if dist < 0.01:
                n_vec = np.array([0., 1.])
            else:
                n_vec = vec / dist
            
            # Constraint: n_vec' * (pos - obs_pos) >= radius - slack
            # Note: We access slack[obs_idx, k]
            constraints += [
                n_vec @ (x[0:2, k] - o_pos) >= o_rad - slack[obs_idx, k]
            ]

    # Trust Region
    for k in range(N):
        constraints += [cp.norm(x[:, k] - x_ref[:, k]) <= 1.5]

    # Actuator Limits
    for k in range(N-1):
        constraints += [cp.norm(u[:, k]) <= 4.0]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS) # ECOS is standard for SOCP

    if x.value is None:
        print("Solver Failed!")
        break

    # Metrics
    step_diff = np.linalg.norm(x.value - x_ref)
    total_slack = np.sum(slack.value)
    
    print(f"{iteration+1:<5} | {step_diff:.4f}     | {total_slack:.4f}")

    # Update Reference
    x_ref = x.value.copy()

    # --- PLOTTING ---
    plt.clf()
    
    # Draw Obstacles
    for obs in obstacles:
        circle = plt.Circle(obs[0:2], obs[2], color='r', alpha=0.3)
        plt.gca().add_patch(circle)

    # Draw Trajectory
    plt.plot(x_ref[0, :], x_ref[1, :], 'b.-', linewidth=2, label='Trajectory')
    plt.plot(p_start[0], p_start[1], 'go', markersize=10, label='Start')
    plt.plot(p_goal[0], p_goal[1], 'rx', markersize=10, markeredgewidth=2, label='Goal')

    # Highlight Violations (Red Stars)
    # Check where slack is active (> 1e-3)
    # slack.value is (n_obs, N), sum along axis 0 to see if ANY obstacle was hit at time k
    slack_per_step = np.sum(slack.value, axis=0)
    violation_indices = np.where(slack_per_step > 1e-3)[0]
    
    if len(violation_indices) > 0:
        plt.plot(x_ref[0, violation_indices], x_ref[1, violation_indices], 
                 'r*', markersize=8, label='Collision (Slack Active)')

    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-4, 4)
    plt.title(f"Iteration {iteration+1} | Total Slack: {total_slack:.2f}")
    plt.legend(loc='upper left')
    plt.pause(0.1)

    if step_diff < 0.1 and total_slack < 1e-3:
        print("Converged Successfully!")
        break

plt.show()