import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- 1. SETUP ---
N = 40                  
dt = 0.2                
r_obs = 1.5             
p_obs = np.array([0., 0.])     
p_start = np.array([-2.5, 0.]) 
p_goal = np.array([2.5, 0.])   

# Dynamics (2D Double Integrator)
A = np.eye(4)
A[0, 2] = dt; A[1, 3] = dt
B = np.zeros((4, 2))
B[0, 0] = 0.5 * dt**2; B[1, 1] = 0.5 * dt**2
B[2, 0] = dt; B[3, 1] = dt

# --- 2. INITIAL GUESS (The Fix: Sine Wave) ---
# We CANNOT use a straight line. We must give the solver a hint 
# to go "around" the obstacle.
x_ref = np.zeros((4, N))
for k in range(N):
    alpha = k / (N - 1)
    # 1. Base Line
    pos_base = (1 - alpha) * p_start + alpha * p_goal
    # 2. Add Sine Wave "Bump" (Amplitude 2.0 to clear obstacle)
    # This breaks symmetry and prevents the solver from getting stuck.
    bump = np.array([0.0, 2.0 * np.sin(np.pi * alpha)])
    
    x_ref[0:2, k] = pos_base + bump

# --- 3. SCP OPTIMIZATION LOOP ---
max_iters = 15
plt.figure(figsize=(10, 8))

for iteration in range(max_iters):
    # Variables
    x = cp.Variable((4, N))
    u = cp.Variable((2, N-1))
    slack = cp.Variable(N, nonneg=True)

    # Cost: Minimize Control + HEAVY Slack Penalty
    cost = cp.sum_squares(u) + 100000 * cp.sum(slack)
    constraints = []
    
    # Dynamics
    constraints += [x[:, 0] == np.hstack([p_start, 0, 0])]
    for k in range(N-1):
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

    # Terminal
    constraints += [cp.norm(x[0:2, N-1] - p_goal) <= 0.1]
    constraints += [x[2:4, N-1] == 0]

    # --- Linearized Obstacle Avoidance ---
    for k in range(1, N-1):
        p_ref = x_ref[0:2, k]
        vec = p_ref - p_obs
        dist = np.linalg.norm(vec)
        
        # Robust Normal Vector
        if dist < 1e-3: 
            n_vec = np.array([0., 1.]) 
        else:
            n_vec = vec / dist
            
        # Constraint: n_vec^T * (p - p_obs) >= Radius - Slack
        constraints += [n_vec @ (x[0:2, k] - p_obs) >= r_obs - slack[k]]

    # Trust Region
    # Crucial: Forces solver to improve LOCALLY around the sine wave
    for k in range(N):
        constraints += [cp.norm(x[:, k] - x_ref[:, k]) <= 1.0]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)

    if x.value is None:
        print("Solver Failed")
        break

    # Convergence Check
    diff = np.linalg.norm(x.value - x_ref)
    max_slack = np.max(slack.value)
    print(f"Iter {iteration+1} | Diff: {diff:.4f} | Max Slack: {max_slack:.4f}")

    x_ref = x.value.copy()

    # --- Plotting ---
    plt.clf()
    circle = plt.Circle(p_obs, r_obs, color='r', alpha=0.3)
    plt.gca().add_patch(circle)
    plt.plot(x_ref[0, :], x_ref[1, :], 'b.-', linewidth=2, label='Trajectory')
    plt.scatter(p_start[0], p_start[1], c='g', s=100, label='Start')
    plt.scatter(p_goal[0], p_goal[1], c='k', s=100, label='Goal')
    plt.xlim(-3, 3); plt.ylim(-2, 3); plt.grid(True)
    plt.title(f"Iteration {iteration+1}")
    plt.pause(0.01)

    if diff < 0.1 and max_slack < 1e-3:
        print("Converged!")
        break

plt.show()