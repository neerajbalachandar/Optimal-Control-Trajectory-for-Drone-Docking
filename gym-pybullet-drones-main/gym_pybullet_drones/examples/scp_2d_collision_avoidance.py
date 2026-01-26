import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
N = 40
dt = 0.1

p_start = np.array([-1.0, 0.0])
p_target = np.array([0.0, 0.0])

r_c = 0.1
r_t = 0.1
alpha_min = 1.0
psi = 50.0

# Dynamics: 2D double integrator
A = np.eye(4)
A[0,2] = dt; A[1,3] = dt
B = np.zeros((4,2))
B[0,0] = 0.5*dt**2
B[1,1] = 0.5*dt**2
B[2,0] = dt
B[3,1] = dt

# -----------------------------
# Initial reference trajectory
# -----------------------------
x_ref = np.zeros((4, N))
for k in range(N):
    a = k/(N-1)
    x_ref[0:2,k] = (1-a)*p_start + a*p_target

# -----------------------------
# SCP iteration (single)
# -----------------------------
x = cp.Variable((4, N))
u = cp.Variable((2, N-1))
slack = cp.Variable(N, nonneg=True)

cost = cp.sum_squares(u) + psi * cp.sum(slack)
constraints = []

constraints += [x[:,0] == np.hstack([p_start, np.zeros(2)])]

for k in range(N-1):
    constraints += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]

constraints += [
    cp.norm(x[0:2,-1] - p_target) <= 0.05,
    x[2:4,-1] == 0
]

# -----------------------------
# Linearized DCOL constraints
# -----------------------------
for k in range(N):
    pref = x_ref[0:2,k]
    d = np.linalg.norm(pref - p_target)

    if d < 1e-6:
        n = np.array([1.0, 0.0])
    else:
        n = (pref - p_target) / d

    alpha_bar = d / (r_c + r_t)
    J = n / (r_c + r_t)

    constraints += [
        alpha_bar + J @ (x[0:2,k] - pref) + slack[k] >= alpha_min
    ]

# -----------------------------
# Trust region
# -----------------------------
for k in range(N):
    constraints += [cp.norm(x[:,k] - x_ref[:,k]) <= 0.8]

# -----------------------------
# Solve
# -----------------------------
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.ECOS)

# -----------------------------
# Plot
# -----------------------------
traj = x.value

plt.figure(figsize=(6,6))
plt.plot(traj[0,:], traj[1,:], 'b.-')
plt.scatter(*p_start, c='g', label="Start")
plt.scatter(*p_target, c='k', label="Target")
plt.gca().add_patch(plt.Circle(p_target, r_t, color='r', alpha=0.3))
plt.gca().add_patch(plt.Circle(traj[0:2,-1], r_c, color='b', alpha=0.3))
plt.axis('equal')
plt.grid()
plt.legend()
plt.title("2D SCP Docking with Linearized DCOL Collision")
plt.show()
