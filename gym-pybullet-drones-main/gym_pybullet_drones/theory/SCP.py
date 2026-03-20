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

# ================= DCOL (Tracy et al.) =================
def alpha(pc):
    """Calculates DCOL scaling factor for the Target"""
    return np.linalg.norm(pc - p_target)/(r_c+r_t)

def grad_alpha(pc):
    """Calculates DCOL analytical gradient to push chaser away"""
    v = pc - p_target
    d = np.linalg.norm(v)+1e-8
    return v/(d*(r_c+r_t))

# ================= INITIAL GUESS =================
x_nom = np.zeros((N,6))
for k in range(N):
    al = k/(N-1)
    x_nom[k,0:3] = (1-al)*x0[0:3] + al*p_target
    
    # THE FIX: We apply a LATERAL bow (Y-axis), not a VERTICAL bow (Z-axis). 
    # This forces the SCP linearization planes to be vertical walls, not horizontal floors!
    x_nom[k,1] += 0.8 * np.sin(np.pi*al) 

# ================= SCP LOOP =================
trust_radius = 2.0
TOL = 1e-3
MAX_ITERS = 15

print("Starting SCP...")
for it in range(MAX_ITERS):
    x = cp.Variable((N,6))
    u = cp.Variable((N-1,3))
    
    cost = 0
    con = [ x[0,:] == x0 ]
    
    # Terminal Boundary Condition (Must reach target at end)
    con += [ x[-1, 0:3] == p_target ]
    con += [ x[-1, 3:6] == np.zeros(3) ]
    
    for k in range(N-1):
        # 1. Dynamics
        con += [ x[k+1,:] == A_d @ x[k,:] + B_d @ u[k,:] ]
        
        # Objective: Tracking Error + Control Effort
        cost += cp.sum_squares(x[k,0:3] - p_target) * 10.0
        cost += cp.sum_squares(u[k,:]) * 1.0
        cost += cp.sum_squares(x[k,3:6]) * 1.0 
        
        # 2. Trust Region
        con += [ cp.norm(x[k,:] - x_nom[k,:], np.inf) <= trust_radius ]
        
        # 3. Obstacle Avoidance (Linearized)
        p_nom = x_nom[k, 0:3]
        v_obs = p_nom - P_OBS
        d_obs = np.linalg.norm(v_obs) + 1e-8
        n_obs = v_obs / d_obs
        con += [ n_obs @ (x[k,0:3] - P_OBS) >= R_OBS + R_SAFE ]

        # 4. Target DCOL Avoidance (Don't crash into target before terminal step)
        if k < N-2: 
            n_tar = grad_alpha(p_nom) * (r_c + r_t) # Denormalizing gradient for standard distance
            # DCOL Linearized Constraint
            con += [ n_tar @ (x[k,0:3] - p_target) >= (r_c + r_t)*alpha_min ]
            
        # 5. Physical Limits
        con += [ cp.norm(u[k,:], np.inf) <= U_MAX ]
        con += [ cp.norm(x[k+1,3:6], 2) <= V_MAX ]

    # Terminal Trust Region
    con += [ cp.norm(x[-1,:] - x_nom[-1,:], np.inf) <= trust_radius ]

    prob = cp.Problem(cp.Minimize(cost), con)
    prob.solve(solver=cp.ECOS)
    
    if prob.status != "optimal":
        print(f"Iter {it}: Infeasible! Shrinking trust region.")
        trust_radius *= 0.5
        continue
        
    delta = np.linalg.norm(x.value - x_nom, np.inf)
    print(f"Iter {it} | delta: {delta:.4f} | trust: {trust_radius:.4f}")
    
    x_nom = x.value.copy()
    
    if delta < TOL:
        print(">>> SCP CONVERGED SUCESSFULLY <<<")
        break
        
    trust_radius = np.clip(1.1 * trust_radius, 0.1, 3.0) # Expand trust slightly if successful

# ================= PLOTTING =================
traj = x_nom[:,0:3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj[:,0], traj[:,1], traj[:,2], 'b.-', linewidth=3, label='Optimized Chaser Traj')
ax.plot(x0[0], x0[1], x0[2], 'go', markersize=8, label='Start')
ax.plot(p_target[0], p_target[1], p_target[2], 'r*', markersize=12, label='Target')

# Plot Obstacle
u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = P_OBS[0] + R_OBS*np.cos(u_sph)*np.sin(v_sph)
y_sph = P_OBS[1] + R_OBS*np.sin(u_sph)*np.sin(v_sph)
z_sph = P_OBS[2] + R_OBS*np.cos(v_sph)
ax.plot_surface(x_sph, y_sph, z_sph, color='r', alpha=0.3, label='Obstacle')

# Plot Target Safety Bubble (DCOL visualization)
x_tar = p_target[0] + (r_c+r_t)*np.cos(u_sph)*np.sin(v_sph)
y_tar = p_target[1] + (r_c+r_t)*np.sin(u_sph)*np.sin(v_sph)
z_tar = p_target[2] + (r_c+r_t)*np.cos(v_sph)
ax.plot_surface(x_tar, y_tar, z_tar, color='g', alpha=0.2, label='Target Safety Zone')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('DCOL-Enforced Drone Trajectory')
ax.legend()
plt.show()