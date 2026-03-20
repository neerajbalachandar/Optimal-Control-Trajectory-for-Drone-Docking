import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ================= SYSTEM =================
dt = 0.1
N = 25

U_MAX = 15.0
V_MAX = 5.0

P_OBS = np.array([-1.0,0.0,1.25])
R_OBS = 0.4
R_SAFE = 0.1

THETA = np.radians(30)

# approach from ABOVE
N_APP = np.array([0,0,-1])

r_c = 0.1
r_t = 0.1
alpha_min = 1.05

x0 = np.array([-2.5,0,1.5,0,0,0])
p_target = np.array([0.5,0,1.0])

# ================= DYNAMICS =================
A_d = np.eye(6)
A_d[0:3,3:6] = dt*np.eye(3)

B_d = np.zeros((6,3))
B_d[0:3,:] = 0.5*dt**2*np.eye(3)
B_d[3:6,:] = dt*np.eye(3)

# ================= DCOL =================
def alpha(pc):
    return np.linalg.norm(pc - p_target)/(r_c+r_t)

def grad_alpha(pc):
    v = pc - p_target
    d = np.linalg.norm(v)+1e-8
    return v/(d*(r_c+r_t))

# ================= OBSTACLE =================
def obs_halfspace(pc):
    v = pc - P_OBS
    d = np.linalg.norm(v)+1e-8
    n = v/d
    d_safe = (R_OBS+R_SAFE) - d
    return n, d_safe

# ================= SCP =================
x_nom = np.tile(x0,(N+1,1))

MAX_SCP = 30
TOL = 1e-2
trust_radius = 3.0

for scp_iter in range(MAX_SCP):

    x = cp.Variable((N+1,6))
    u = cp.Variable((N,3))
    s_cone = cp.Variable(N)

    constraints = [x[0]==x0, s_cone >= 0]
    cost = 0

    for k in range(N):

        constraints += [x[k+1] == A_d@x[k] + B_d@u[k]]

        constraints += [cp.norm_inf(u[k]) <= U_MAX]
        constraints += [cp.norm(x[k,3:6]) <= V_MAX]
        constraints += [cp.norm(x[k,0:3] - x_nom[k,0:3]) <= trust_radius]

        # obstacle
        n_obs,d_safe = obs_halfspace(x_nom[k,0:3])
        constraints += [n_obs @ (x[k,0:3]-x_nom[k,0:3]) >= d_safe]

        # ===== SOFT DOCKING CONE =====
        p_rel_nom = x_nom[k,0:3] - p_target
        norm_nom = np.linalg.norm(p_rel_nom)+1e-8

        f_k = -N_APP @ p_rel_nom - norm_nom*np.cos(THETA)
        grad_f = -N_APP - (p_rel_nom/norm_nom)*np.cos(THETA)

        constraints += [
            f_k + grad_f@(x[k,0:3]-x_nom[k,0:3]) >= -s_cone[k]
        ]

             # ===== SOFT CONE PENALTY =====
        cost += 200 * s_cone[k]

        # ===== DCOL near terminal =====
        if norm_nom < 0.5:
            a_k = alpha(x_nom[k,0:3])
            g_k = grad_alpha(x_nom[k,0:3])
            constraints += [
                a_k + g_k @ (x[k,0:3] - x_nom[k,0:3]) >= alpha_min
            ]


        cost += 0.3 * cp.square(x[k,2] - p_target[2])  #unnecessary vertical climb

        # ===== POSITION TRACKING =====
        cost += 2.0 * cp.sum_squares(x[k,0:3] - p_target)

        # ===== VELOCITY DAMPING =====
        cost += 0.5 * cp.sum_squares(x[k,3:6])

        # ===== CONTROL EFFORT =====
        cost += 0.05 * cp.sum_squares(u[k])

    # ===== HARD TERMINAL CONE =====
    p_rel_T = x[N,0:3] - p_target
    constraints += [
        -N_APP @ p_rel_T >= cp.norm(p_rel_T)*np.cos(THETA)
    ]
    
    cost += 50 * cp.sum_squares(x[N,0:3] - p_target)
    cost += 20 * cp.sum_squares(x[N,3:6])
    cost += 5 * cp.sum_squares(x[N,0:3] - p_target)
    cost += 3 * cp.sum_squares(x[N,3:6])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)

    print("Iter",scp_iter,"status:",prob.status)

    if prob.status != "optimal":
        trust_radius *= 0.5
        continue

    delta = np.linalg.norm(x.value - x_nom)
    print("traj change:",delta)
    print("terminal:", x.value[-1,0:3])

    if delta < TOL:
        print("SCP CONVERGED")
        x_nom = x.value.copy()
        break

    x_nom = x.value.copy()
    trust_radius = 0.9*trust_radius + 0.1*delta
    trust_radius = np.clip(trust_radius,0.3,5.0)

print("Final position:", x_nom[-1,0:3])
print("Final alpha:", alpha(x_nom[-1,0:3]))

traj = x_nom[:,0:3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ===== trajectory =====
ax.plot(traj[:,0], traj[:,1], traj[:,2], linewidth=3, label='Chaser Traj')

# ===== sphere plotting helper =====
def plot_sphere(ax, center, radius, color, alpha=0.3):
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius*np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius*np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius*np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

# ===== obstacle sphere =====
plot_sphere(ax, P_OBS, R_OBS, 'black', alpha=0.4)

# ===== target sphere =====
plot_sphere(ax, p_target, 0.1, 'red', alpha=0.6)

# ===== start point =====
ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='blue', s=80, label='Start')

# ===== formatting =====
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_box_aspect([1,1,1])   # IMPORTANT: true geometry

ax.legend()
plt.show()