import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def omega_coeffs(z, omega0, phi, zp, nn):
    """
    For a given propagation coordinate z, compute the effective 
    coefficients omega1 and omega2 using a piecewise-constant model.
    
    In the Mathematica code the sum over m (from 1 to nn) with
      UnitStep[z - sum_{k=1}^m zp[k]] - UnitStep[z - sum_{k=1}^{m+1} zp[k]]
    selects the mth segment.
    
    Here we compute the cumulative sums of zp (which is assumed to have length nn+1)
    and check in which segment z falls. Then, omega1 and omega2 are given by:
    
       omega1 = omega0 * sin(phi[m-1])
       omega2 = omega0 * cos(phi[m-1])
    
    (Only one term will contribute if z is not exactly at a boundary.)
    """
    # Compute cumulative distances; note that zp is given as a list/array.
    c = np.cumsum(zp)
    omega1 = 0.0
    omega2 = 0.0
    # Loop over segments m = 1,...,nn (Mathematica indexing)
    for m in range(1, nn + 1):
        lower = c[m - 1]
        upper = c[m]  # since len(zp)==nn+1, c[m] exists for m=1,...,nn.
        if (z >= lower) and (z < upper):
            omega1 += omega0 * np.sin(phi[m - 1])
            omega2 += omega0 * np.cos(phi[m - 1])
    return omega1, omega2

def ode_system(z, y, omega0, omega3, zp, phi, nn):
    """
    Defines the system of three coupled ODEs:
      E1' = -i * omega1 * E2 - omega3 * E3
      E2' = -i * conjugate(omega1) * E1 - i * Delta * E2 - i * conjugate(omega2) * E3
      E3' =  omega3 * E1 - i * omega2 * E2
    with Delta = 0.
    
    The coefficients omega1 and omega2 depend on z through a piecewise function.
    """
    E1, E2, E3 = y
    omega1, omega2 = omega_coeffs(z, omega0, phi, zp, nn)
    delta = 0.0  # as in the Mathematica code
    dE1dz = -1j * omega1 * E2 - omega3 * E3
    dE2dz = -1j * np.conjugate(omega1) * E1 - 1j * delta * E2 - 1j * np.conjugate(omega2) * E3
    dE3dz = omega3 * E1 - 1j * omega2 * E2
    return [dE1dz, dE2dz, dE3dz]

def Population(omega0, omega3, zp, phi, zmin, zmax, nn):
    """
    Solves the ODE system from zmin to zmax with initial conditions
      E1(zmin) = 1, E2(zmin) = 0, E3(zmin) = 0
    and returns |E3(zmax)|^2.
    """
    # Initial conditions (using complex numbers)
    y0 = [1+0j, 0+0j, 0+0j]
    sol = solve_ivp(fun=lambda z, y: ode_system(z, y, omega0, omega3, zp, phi, nn),
                    t_span=(zmin, zmax), y0=y0, method='RK45',
                    atol=1e-8, rtol=1e-6)
    # Extract E3 at z = zmax (the last computed point)
    E3_final = sol.y[2, -1]
    return np.abs(E3_final)**2

# ------------------------------------------------------------
# Parameters (using the second set from your code)
Eta = 1.0
# zp has length nn+1; here for nn=10 we need 11 elements.
zp = [0, Eta, Eta, Eta, Eta, Eta, Eta, Eta, Eta, Eta, Eta]
nn = 10
Alpha = np.pi / 2

# Final definition of phi (length nn = 10):
phi = [0,
       5 * np.pi / 6,
       np.pi / 3,
       5 * np.pi / 6,
       0,
       np.pi + Alpha/2,
       11 * np.pi/6 + Alpha/2,
       4 * np.pi/3 + Alpha/2,
       11 * np.pi/6 + Alpha/2,
       np.pi + Alpha/2]

zmin = 0.0
zmax = nn * Eta  # total propagation length (10)
zstep = 0.1     # (not directly used in integration)

# Define the ranges for omega3 and omega0.
# In the Mathematica code, omega3 goes from 0 to 5 with step 5/20, and
# omega0 goes from 5 to 0 with step -5/20.
omega3_min = 0.0
omega3_max = 5.0
omega3_step = 5.0 / 20.0

omega0_max = 0.0
omega0_min = 5.0
omega0_step = 5.0 / 20.0  # descending order

omega0_values = np.arange(omega0_max, omega0_min - 1e-6, omega0_step)
omega3_values = np.arange(omega3_min, omega3_max + 1e-6, omega3_step)

# Prepare a grid to hold the computed Population values P3.
P3 = np.zeros((len(omega0_values), len(omega3_values)))

# Loop over the grid and compute Population for each (omega0, omega3) pair.
for i, omega0 in enumerate(omega0_values):
    for j, omega3 in enumerate(omega3_values):
        P3[i, j] = Population(omega0, omega3, zp, phi, zmin, zmax, nn)

# Plot a contour map of P3.
# Note: meshgrid is used with omega3 on the horizontal axis and omega0 on the vertical.
Omega3_grid, Omega0_grid = np.meshgrid(omega3_values, omega0_values)
plt.contourf(Omega3_grid, Omega0_grid, P3, levels=50, cmap='viridis')
plt.xlabel(r'$\Omega_3$')
plt.ylabel(r'$\Omega_0$')
plt.title(r'$P_3 = |E_3(z_{max})|^2$')
plt.colorbar(label=r'$P_3$')
plt.show()

# # Export the data to a file (using a Windows-style path as in your code).
# np.savetxt("c:\\data\\P3.dat", P3)
