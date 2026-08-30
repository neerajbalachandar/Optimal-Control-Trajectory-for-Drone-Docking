import numpy as np

# =====================================================================
# SYSTEM PARAMETERS (Derived from the paper's theory)
# =====================================================================
dt = 0.1                      # Time step (\Delta t)
U_MAX = 15.0                  # Infinity-norm control bound (Eq 9)
V_MAX = 5.0                   # L2-norm velocity bound (Eq 10)

# Obstacle Parameters
P_OBS = np.array([-1.0, 0.0, 1.25])  # Position of obstacle
R_OBS = 0.4                          # Radius of obstacle
R_SAFE = 0.1                         # Safety margin

# Docking Cone Parameters (Eq 14)
THETA_CONE_DEG = 30.0         
THETA_CONE_RAD = np.radians(THETA_CONE_DEG)
N_APP = np.array([0.0, 0.0, -1.0])   # Docking approach axis (\hat{n}_{app})

print("=== PART 1: STATE SPACE REPRESENTATION ===")

# =====================================================================
# CONTINUOUS TIME DYNAMICS (Eq 1)
# \dot{x} = A_c * x + B_c * u
# =====================================================================
# State x = [px, py, pz, vx, vy, vz]^T \in R^6
# Control u = [ux, uy, uz]^T \in R^3 (Net acceleration)
A_c = np.zeros((6, 6))
A_c[0:3, 3:6] = np.eye(3)  # \dot{p} = v

B_c = np.zeros((6, 3))
B_c[3:6, 0:3] = np.eye(3)  # \dot{v} = u

print("\nContinuous-Time Matrix A_c (6x6):")
print(A_c)
print("\nContinuous-Time Matrix B_c (6x3):")
print(B_c)

# =====================================================================
# DISCRETE TIME DYNAMICS (Eq 2)
# x_{k+1} = A_d * x_k + B_d * u_k
# =====================================================================
# Using Zero-Order Hold discretization:
# A_d = e^{A_c * dt}
# B_d = \int_0^{dt} e^{A_c * \tau} B_c d\tau

A_d = np.eye(6)
A_d[0:3, 3:6] = np.eye(3) * dt

B_d = np.zeros((6, 3))
B_d[0:3, 0:3] = 0.5 * (dt**2) * np.eye(3)
B_d[3:6, 0:3] = dt * np.eye(3)

print(f"\nDiscrete-Time Matrix A_d (dt={dt}s):")
print(A_d)
print(f"\nDiscrete-Time Matrix B_d (dt={dt}s):")
print(B_d)


print("\n\n=== PART 2: DYNAMICS PROPAGATION ===")

# Initialize Drones
# Chaser state: x_c = [px, py, pz, vx, vy, vz]
x_c_k = np.array([-2.5, 0.0, 1.5, 0.0, 0.0, 0.0]) 

# Target state: x_t = [px, py, pz, vx, vy, vz]
x_t_k = np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0])

# Apply a theoretical control input (e.g., accelerating forward and slightly down)
u_k = np.array([2.0, 0.0, -0.5])

print(f"Time k   | Chaser State x_k: {x_c_k}")
print(f"Time k   | Control Input u_k: {u_k}")

# Mathematically propagate to k+1 (Eq 2)
x_c_k1 = A_d @ x_c_k + B_d @ u_k

print(f"Time k+1 | Chaser State x_k+1: {x_c_k1}")


print("\n\n=== PART 3: CONSTRAINT EVALUATION (At time k+1) ===")

# Extract position and velocity from the new state
p_c = x_c_k1[0:3]
v_c = x_c_k1[3:6]
p_t = x_t_k[0:3] # Assuming target is static for this check

# 1. Control Limits (Eq 9): ||u_k||_\infty <= U_max
inf_norm_u = np.linalg.norm(u_k, ord=np.inf)
check_u = inf_norm_u <= U_MAX
print(f"Eq  9 (Control Bound)  : ||u||_inf = {inf_norm_u:.2f} <= {U_MAX} --> {check_u}")

# 2. Velocity Limits (Eq 10): ||v_{c,k}||_2 <= V_max
l2_norm_v = np.linalg.norm(v_c)
check_v = l2_norm_v <= V_MAX
print(f"Eq 10 (Velocity Bound) : ||v||_2   = {l2_norm_v:.2f} <= {V_MAX} --> {check_v}")

# 3. Obstacle Collision Avoidance (Eq 13)
# ||p_c - p_obs||_2 >= r_obs + r_safe
dist_obs = np.linalg.norm(p_c - P_OBS)
min_safe_dist = R_OBS + R_SAFE
check_obs = dist_obs >= min_safe_dist
print(f"Eq 13 (Obstacle Avoid) : Distance  = {dist_obs:.2f} >= {min_safe_dist} --> {check_obs}")

# 4. Approach Cone Constraint (Eq 14)
# The vector from target to chaser must lie within the cone
p_rel = p_c - p_t

# Mathematical formulation: ||p_rel||_2 * cos(theta) <= -(n_app^T * p_rel)
# (Note: Negative sign depends on whether n_app points INTO or OUT OF the cone. 
# Here, n_app=[0,0,-1] points down, so the relative vector pointing UP should have a negative dot product)
lhs_cone = np.linalg.norm(p_rel) * np.cos(THETA_CONE_RAD)
rhs_cone = -np.dot(N_APP, p_rel)

check_cone = lhs_cone <= rhs_cone
print(f"Eq 14 (Approach Cone)  : {lhs_cone:.3f} <= {rhs_cone:.3f} --> {check_cone}")

# Provide insight on the Cone Constraint failure/success
if not check_cone:
    print("   -> INSIGHT: The Chaser is currently outside the admissible docking cone funnel!")
else:
    print("   -> INSIGHT: The Chaser is safely inside the docking cone funnel.")