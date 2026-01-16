import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ----------------------------
# 1. SETUP
# ----------------------------
N = 50
dt = 0.2

# Obstacle (sphere)
p_obs = np.array([0., 0., 0.])
r_obs = 1.5

# Start / Goal
p_start = np.array([-3., 0., 0.])
p_goal  = np.array([ 3., 0., 0.])

# ----------------------------
# 2. 3D DOUBLE INTEGRATOR
# ----------------------------
A = np.eye(6)
A[0, 3] = dt; A[1, 4] = dt; A[2, 5] = dt

B = np.zeros((6, 3))
B[0, 0] = 0.5 * dt**2
B[1, 1] = 0.5 * dt**2
B[2, 2] = 0.5 * dt**2
B[3, 0] = dt
B[4, 1] = dt
B[5, 2] = dt

# ----------------------------
# 3. INITIAL GUESS (3D BUMP)
# ----------------------------
x_ref = np.zeros((6, N))
for k in range(N):
    alpha = k / (N - 1)

    pos_base = (1 - alpha) * p_start + alpha * p_goal

    # 3D bump (go around sphere in y-z plane)
    bump = np.array([
        0.0,
        2.0 * np.sin(np.pi * alpha),
        1.5 * np.sin(np.pi * alpha)
    ])

    x_ref[0:3, k] = pos_base + bump

# ----------------------------
# 4. SCP LOOP
# ----------------------------
max_iters = 20

for iteration in range(max_iters):
    x = cp.Variable((6, N))
    u = cp.Variable((3, N-1))
    slack = cp.Variable(N, nonneg=True)

    cost = cp.sum_squares(u) + 1e5 * cp.sum(slack)
    constraints = []

    # Initial condition
    constraints += [x[:, 0] == np.hstack([p_start, np.zeros(3)])]

    # Dynamics
    for k in range(N-1):
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

    # Terminal
    constraints += [
        cp.norm(x[0:3, -1] - p_goal) <= 0.1,
        x[3:6, -1] == 0
    ]

    # Obstacle linearization
    for k in range(1, N-1):
        p_ref = x_ref[0:3, k]
        vec = p_ref - p_obs
        dist = np.linalg.norm(vec)

        if dist < 1e-3:
            n_vec = np.array([0., 1., 0.])
        else:
            n_vec = vec / dist

        constraints += [
            n_vec @ (x[0:3, k] - p_obs) >= r_obs - slack[k]
        ]

    # Trust region
    for k in range(N):
        constraints += [
            cp.norm(x[:, k] - x_ref[:, k]) <= 1.0
        ]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)

    if x.value is None:
        print("Solver failed")
        break

    diff = np.linalg.norm(x.value - x_ref)
    max_slack = np.max(slack.value)
    print(f"Iter {iteration+1} | Diff: {diff:.4f} | Max Slack: {max_slack:.4e}")

    x_ref = x.value.copy()

    if diff < 0.1 and max_slack < 1e-3:
        print("Converged!")
        break

# ----------------------------
# 5. SIMPLE 3D VISUALIZATION
# ----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Trajectory
ax.plot(x_ref[0, :], x_ref[1, :], x_ref[2, :], 'b.-', label="Trajectory")

# Start / Goal
ax.scatter(*p_start, c='g', s=80, label="Start")
ax.scatter(*p_goal,  c='k', s=80, label="Goal")

# Obstacle (sphere)
u_s, v_s = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs = p_obs[0] + r_obs * np.cos(u_s) * np.sin(v_s)
ys = p_obs[1] + r_obs * np.sin(u_s) * np.sin(v_s)
zs = p_obs[2] + r_obs * np.cos(v_s)
ax.plot_wireframe(xs, ys, zs, color="r", alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("3D SCP Trajectory with Obstacle Avoidance")
plt.show()
