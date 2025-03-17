import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

# Set simulation parameters
tmin = 0
tmax = 10
tstep = 0.01
time_points = np.arange(tmin, tmax + tstep, tstep)

# Define the coupling constants
Omega1 = 4
Omega2 = 4
Omega3 = 1.3
Delta = 0

# Define the Hamiltonian matrix (time-independent)
H = np.array([[0,            Omega1,       -1j * Omega3],
              [Omega1,       0,             Omega2],
              [1j * Omega3,  Omega2,        0]], dtype=complex)

# Define the differential equation: dy/dt = -i * H * y
def dydt(t, y):
    return -1j * H.dot(y)

# Initial condition: all population in state 1
y0 = np.array([1, 0, 0], dtype=complex)

# Solve the ODE system
solution = solve_ivp(dydt, [tmin, tmax], y0, t_eval=time_points, max_step=tstep)

# Compute probabilities for each state: |y_i(t)|^2
P1 = np.abs(solution.y[0, :])**2
P2 = np.abs(solution.y[1, :])**2
P3 = np.abs(solution.y[2, :])**2

# Compute cumulative probabilities for stacking
cumP1 = P1
cumP2 = P1 + P2
cumP3 = P1 + P2 + P3  # Should sum to 1 at each time point

# Create a figure for the stacked area plot using fill_between
plt.figure(figsize=(10, 6))

# First area: from 0 to cumP1
plt.fill_between(solution.t, 0, cumP1, 
                 facecolor='0.9', hatch='///', edgecolor='k', linewidth=0.5)

# Second area: from cumP1 to cumP2
plt.fill_between(solution.t, cumP1, cumP2, 
                 facecolor='0.7', hatch='\\\\', edgecolor='k', linewidth=0.5)

# Third area: from cumP2 to cumP3 (which should be 1)
plt.fill_between(solution.t, cumP2, cumP3, 
                 facecolor='0.5', hatch='xxx', edgecolor='k', linewidth=0.5)

plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Population Dynamics for a Single Pulse')
plt.xlim(tmin, tmax)
plt.ylim(0, 1)
plt.grid(True)

# Create custom legend patches
legend_handles = [
    Patch(facecolor='0.9', hatch='///', edgecolor='k', label='P1'),
    Patch(facecolor='0.7', hatch='\\\\', edgecolor='k', label='P2'),
    Patch(facecolor='0.5', hatch='xxx', edgecolor='k', label='P3')
]

# Make some room on the right for the legend
plt.subplots_adjust(right=0.9)

# Place a single legend for the whole figure, anchored outside on the right.
plt.legend(
    handles=legend_handles[::-1],
    loc='center right',       # vertically center on the right edge
    bbox_to_anchor=(1.1, 0.8),
    borderaxespad=0,
    labelspacing=5.0          # spread out entries vertically
)


plt.show()
