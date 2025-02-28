import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the effective Rabi frequency Omega(z)
def omega_of_z(z, Omega0, zp, Phi):
    """
    Returns the effective complex Rabi frequency at position z.
    The composite pulse is built as a piecewise constant function:
      Omega(z) = Omega0 * exp(i * Phi[m])
    for z in [cumsum[m-1], cumsum[m]), m = 1,..., nn.
    
    Parameters:
      z      : propagation coordinate (float)
      Omega0 : amplitude (float)
      zp     : list or array of segment lengths (length = nn+1)
      Phi    : list or array of phases (length = nn)
    """
    # Compute cumulative boundaries
    cumsum = np.cumsum(zp)
    # Find the segment index m such that cumsum[m-1] <= z < cumsum[m]
    m = np.searchsorted(cumsum, z, side='right')
    if m < 1 or m > len(Phi):
        # If z is outside the defined segments, return zero.
        return 0.0
    else:
        return Omega0 * np.exp(1j * Phi[m - 1])

# Define the ODE system for the two-state system
def ode_system(z, y, Omega0, Delta, zp, Phi):
    """
    y[0] = E1(z), y[1] = E2(z)
    
    The equations:
      dE1/dz = i*Delta*E1 - i*Omega(z)*E2
      dE2/dz = -i*Delta*E2 - i*conjugate(Omega(z))*E1
    """
    # Compute the effective Rabi frequency at z
    Omega = omega_of_z(z, Omega0, zp, Phi)
    E1, E2 = y
    dE1dz = 1j * Delta * E1 - 1j * Omega * E2
    dE2dz = -1j * Delta * E2 - 1j * np.conjugate(Omega) * E1
    return [dE1dz, dE2dz]

def Population(Omega0, Delta, zp, Phi, zmin=0.0, zmax=5.0):
    """
    Solves the two-state Schrödinger equation from zmin to zmax and
    returns |E2(zmax)|^2.
    """
    # Initial conditions: E1(zmin)=1, E2(zmin)=0
    y0 = [1+0j, 0+0j]
    sol = solve_ivp(fun=lambda z, y: ode_system(z, y, Omega0, Delta, zp, Phi),
                    t_span=(zmin, zmax), y0=y0, method='RK45',
                    t_eval=[zmax], max_step=0.01, atol=1e-8, rtol=1e-6)
    E2_final = sol.y[1, -1]
    return np.abs(E2_final)**2

# -----------------------------------------------------------
# Define simulation parameters

# Propagation interval
zmin = 0.0
zmax = 5.0  # total length (should match cumulative sum of zp)

# Parameter ranges for Delta and Omega0
Delta_min = -2.0
Delta_max = 2.0
Delta_step = 4.0 / 100  # 0.04

Omega0_max = 3.0
Omega0_min = 0.0
Omega0_step = -3.0 / 100  # descending from 3 to 0

# Create arrays for Omega0 and Delta values
Omega0_values = np.arange(Omega0_max, Omega0_min - 1e-6, Omega0_step)
Delta_values = np.arange(Delta_min, Delta_max + 1e-6, Delta_step)

# Define composite pulse parameters
Eta = 1.0
# zp: list of segment lengths. For nn = 5, we need 6 numbers.
zp = [0, Eta, Eta, Eta, Eta, Eta]  # cumulative sum gives boundaries: 0, 1, 2, 3, 4, 5
nn = 5
# Phi: phases for each segment (in radians)
Phi = [0, 5*np.pi/6, np.pi/3, 5*np.pi/6, 0]
# (An alternative test would be: Phi = [0, 0, 0, 0, 0])

# Initialize grid for the final population |E2(zmax)|^2 (P1)
P1 = np.zeros((len(Omega0_values), len(Delta_values)))

# Loop over the grid of parameters
for i, Omega0 in enumerate(Omega0_values):
    for j, Delta in enumerate(Delta_values):
        P1[i, j] = Population(Omega0, Delta, zp, Phi, zmin, zmax)

# Create meshgrid for plotting: horizontal axis Delta, vertical axis Omega0
Delta_grid, Omega0_grid = np.meshgrid(Delta_values, Omega0_values)

# Plot contour of the final population P1
plt.figure(figsize=(8,6))
contour = plt.contourf(Delta_grid, Omega0_grid, P1, levels=50, cmap='viridis')
plt.xlabel(r'$\Delta$')
plt.ylabel(r'$\Omega_0$')
plt.title(r'$P_1=|E_2(z_{\max})|^2$')
plt.colorbar(contour, label=r'$P_1$')
plt.show()
