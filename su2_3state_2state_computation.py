import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the composite pulse parameters
# -------------------------------

Eta = 1.0
# zp: list of segment boundaries (length = nn+1)
zp = [0, Eta, Eta, Eta, Eta, Eta]  # For nn = 5 segments; cumulative boundaries: 0,1,2,3,4,5.
nn = 5
# Phi: composite pulse phases (in radians) for each segment (length = nn)
Alpha = np.pi / 2

# Final definition of phi (length nn = 10):
Phi = [0,
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
zmax = nn * Eta  # total propagation length, here 5

# -------------------------------
# 2. Helper: Determine the phase at a given z
# -------------------------------
def phi_eff(z, zp, Phi):
    """
    Returns the composite phase value for the effective two-state system at position z.
    zp: list of segment lengths (or boundaries) such that the cumulative sum defines the intervals.
    Phi: list of phases for each segment.
    """
    # Compute cumulative boundaries (excluding the first element, which is zero)
    boundaries = np.cumsum(zp)
    # Find the segment index m (1-indexed) such that boundaries[m-1] <= z < boundaries[m]
    m = np.searchsorted(boundaries, z, side='right')
    if m < 1 or m > len(Phi):
        # If z is outside the defined segments, return 0.
        return 0.0
    else:
        return Phi[m - 1]

# -------------------------------
# 3. Define the effective two-state ODE system
# -------------------------------
def effective_ode(z, d, Omega0, Omega3, zp, Phi):
    """
    ODE system for the effective two-state system.
    d = [d1, d2] is the state vector.
    
    The effective Hamiltonian is:
      H_eff(z) = 1/2 * [[Omega3, -i*Omega0*exp(i*phi_eff(z))],
                        [i*Omega0*exp(-i*phi_eff(z)), -Omega3]]
    The Schrödinger equation in the effective two-state picture is:
      i d/dz d = H_eff(z) d,
    so that
      d/dz d = -i H_eff(z) d.
    """
    phi_val = phi_eff(z, zp, Phi)
    # Off-diagonal elements
    off_diag = Omega0 * np.exp(1j * phi_val)
    # Construct the Hamiltonian H_eff(z)
    H_eff = 0.5 * np.array([[Omega3, -1j * off_diag],
                            [1j * np.conjugate(off_diag), -Omega3]])
    # Compute derivative: d' = -i * H_eff * d
    d_vec = np.array(d)
    dprime = -1j * H_eff.dot(d_vec)
    return dprime

# -------------------------------
# 4. Function to compute the three-state |3> population using the effective two-state system
# -------------------------------
def Population_3_from_2state(Omega0, Omega3, zp, Phi, zmin=0.0, zmax=5.0):
    """
    Solves the effective two-state system from zmin to zmax with the given parameters.
    Then extracts the Cayley–Klein parameters:
       a = d1(zmax),   b = -conjugate(d2(zmax))
    and computes the three-state propagator element U13 via:
       U13 = (i/2) [b^2 - a^2 + a*^2 - (b*)^2]
    Finally returns P3 = |U13|^2.
    """
    # Initial condition for the effective two-state system: d(0) = [1, 0]
    d0 = [1+0j, 0+0j]
    sol = solve_ivp(fun=lambda z, d: effective_ode(z, d, Omega0, Omega3, zp, Phi),
                    t_span=(zmin, zmax), y0=d0, method='RK45', t_eval=[zmax],
                    atol=1e-8, rtol=1e-6)
    # Extract the final state at zmax
    d_final = sol.y[:, -1]
    a = d_final[0]
    # From the two-state propagator, we have d(zmax) = [a, -b*]^T so that b = -conjugate(d2)
    b = -np.conjugate(d_final[1])
    
    # Now, using the Carroll-Hioe mapping the (1,3) element of the 3-state propagator is:
    U13 = (1j/2.0) * (b**2 - a**2 + np.conjugate(a)**2 - np.conjugate(b)**2)
    P3 = np.abs(U13)**2
    return P3

# -------------------------------
# 5. Sweep parameters and plot the results
# -------------------------------

# Define parameter ranges for Omega0 and Omega3
Omega0_max = 5.0
Omega0_min = 0.0
n_Omega0 = 500
Omega0_values = np.linspace(Omega0_max, Omega0_min, n_Omega0)

Omega3_min = 0.0
Omega3_max = 5.0
n_Omega3 = 500
Omega3_values = np.linspace(Omega3_min, Omega3_max, n_Omega3)

# Prepare grid for population P3
P3_grid = np.zeros((len(Omega0_values), len(Omega3_values)))

for i, Omega0 in enumerate(Omega0_values):
    for j, Omega3 in enumerate(Omega3_values):
        P3_grid[i, j] = Population_3_from_2state(Omega0, Omega3, zp, Phi, zmin, zmax)

# Create meshgrid for plotting: horizontal axis Omega3, vertical axis Omega0
Omega3_grid, Omega0_grid = np.meshgrid(Omega3_values, Omega0_values)

plt.figure(figsize=(8,6))
contour = plt.contourf(Omega3_grid, Omega0_grid, P3_grid, levels=50, cmap='viridis')
plt.xlabel(r'$\Omega_3$')
plt.ylabel(r'$\Omega_0$')
plt.title(r'$P_3 = |U^{(3)}_{13}|^2$ (from effective 2-state system)')
plt.colorbar(contour, label=r'$P_3$')
plt.savefig('population_transfer_3state_su2_compute.png')
plt.show()
