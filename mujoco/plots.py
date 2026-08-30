import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Create folder for saving
os.makedirs("ieee_plots", exist_ok=True)

# Enforce IEEE Publication Standards
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#b0b0b0',
    'axes.titlesize': 0
})

SINGLE_COL = (4.0, 3.2)

# =========================================================
# EXPERIMENT 1: Approach Angle vs. Time to Dock
# =========================================================
# Sorted data for clean line plotting
angles = np.array([10, 20, 30, 40])
times_angle = np.array([7.6, 6.9, 5.68, 4.9])

fig1 = plt.figure(figsize=SINGLE_COL)
ax1 = fig1.add_subplot(111)

# Plotting with square markers
ax1.plot(angles, times_angle, 'bs-', markersize=8, linewidth=2, label='Docking Time')

ax1.set_xlabel('Docking Cone Angle (deg)')
ax1.set_ylabel('Time to Dock (s)')
ax1.set_xticks(angles)
ax1.set_ylim([4.0, 8.5])

# Add a subtle text box to denote conditions
ax1.text(0.95, 0.90, 'Constant Wind: $\sigma=0.1$', 
         transform=ax1.transAxes, ha='right', va='top', 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
fig1.savefig("ieee_plots/experiment_1_cone_angle.png", dpi=300, bbox_inches='tight')

# =========================================================
# EXPERIMENT 2: Wind Disturbance vs. Time to Dock
# =========================================================
winds = np.array([0.1, 0.3, 0.5, 1.0])
times_wind = np.array([5.68, 5.98, 6.7, 10.0]) # 10.0 used to represent timeout boundary

fig2 = plt.figure(figsize=SINGLE_COL)
ax2 = fig2.add_subplot(111)

# Successful region
ax2.plot(winds[:3], times_wind[:3], 'go-', markersize=7, linewidth=2, label='Successful Dock')

# Transition to failure (lighter + thinner)
ax2.plot(winds[2:], times_wind[2:], 'r--', linewidth=1.8, alpha=0.7)

# Failure point (cleaner)
ax2.plot(winds[3], times_wind[3], 'rx', markersize=9, markeredgewidth=2)

# Subtle annotation (no aggressive arrow)
ax2.annotate('Failure (>10s)',
             xy=(1.0, 10.0),
             xytext=(0.75, 10.4),
             textcoords='data',
             fontsize=10,
             ha='center',
             color='red',
             arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

# Labels
ax2.set_xlabel('Wind Standard Deviation (m/s)')
ax2.set_ylabel('Time to Dock (s)')
ax2.set_xticks(winds)
ax2.set_ylim([5.0, 11.0])

# Move legend slightly DOWN and keep it clean
ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85), frameon=True)

# Condition box (less intrusive)
ax2.text(0.97, 0.08, 'Cone Angle: $\\theta=30^\circ$', 
         transform=ax2.transAxes,
         ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.7))

plt.tight_layout()
fig2.savefig("ieee_plots/experiment_2_wind_robustness.png", dpi=300, bbox_inches='tight')

plt.show()