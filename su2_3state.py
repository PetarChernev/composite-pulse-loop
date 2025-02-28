from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

def omega_coeffs(z, omega0, phi, zp_cumsum):
    """
    Compute effective coefficients omega1 and omega2 for a given z,
    using a precomputed cumulative sum of zp.
    
    Parameters:
      z         : current propagation coordinate (scalar)
      omega0    : overall amplitude
      phi       : list/array of phases for each segment (length = nn)
      zp_cumsum : precomputed cumulative sum of zp (length = nn+1)
    
    Returns:
      omega1 = omega0 * sin(phi[idx])
      omega2 = omega0 * cos(phi[idx])
      where idx is determined using np.searchsorted.
    """
    # Find the segment index such that zp_cumsum[idx] <= z < zp_cumsum[idx+1]
    idx = np.searchsorted(zp_cumsum, z, side='right') - 1
    if idx < 0 or idx >= len(phi):
        return 0.0, 0.0
    return omega0 * np.sin(phi[idx]), omega0 * np.cos(phi[idx])

def ode_system(z, y, omega0, omega3, zp_cumsum, phi):
    """
    Defines the system of three coupled ODEs:
      E1' = -i * omega1 * E2 - omega3 * E3
      E2' = -i * conjugate(omega1) * E1 - i * Delta * E2 - i * conjugate(omega2) * E3
      E3' =  omega3 * E1 - i * omega2 * E2
    with Delta = 0.
    
    Uses the efficient version of omega_coeffs.
    """
    E1, E2, E3 = y
    omega1, omega2 = omega_coeffs(z, omega0, phi, zp_cumsum)
    delta = 0.0  # as in the Mathematica code
    dE1dz = -1j * omega1 * E2 - omega3 * E3
    dE2dz = -1j * np.conjugate(omega1) * E1 - 1j * delta * E2 - 1j * np.conjugate(omega2) * E3
    dE3dz = omega3 * E1 - 1j * omega2 * E2
    return [dE1dz, dE2dz, dE3dz]

def Population(omega0, omega3, zp_cumsum, phi, zmin, zmax):
    """
    Solves the ODE system from zmin to zmax with initial conditions
      E1(zmin)=1, E2(zmin)=0, E3(zmin)=0
    and returns |E3(zmax)|^2.
    """
    y0 = [1+0j, 0+0j, 0+0j]
    sol = solve_ivp(lambda z, y: ode_system(z, y, omega0, omega3, zp_cumsum, phi),
                    t_span=(zmin, zmax), y0=y0, method='RK45',
                    atol=1e-8, rtol=1e-6)
    E3_final = sol.y[2, -1]
    return np.abs(E3_final)**2

# Worker function defined at the top level for pickling.
def worker(args):
    omega0, omega3 = args
    # Use the global variables zp_cumsum, phi, zmin, and zmax defined in __main__
    return Population(omega0, omega3, zp_cumsum, phi, zmin, zmax)

if __name__ == '__main__':
    # -------------------------------
    # Parameters (using the second set from your code)
    # -------------------------------
    Eta = 1.0
    nn = 10
    # zp has length nn+1; here we use [0, Eta, Eta, ..., Eta]
    zp = [0] + [Eta] * nn
    # Precompute cumulative sum of zp
    zp_cumsum = np.array(np.cumsum(zp))
    
    Alpha = np.pi / 2
    # Final definition of phi (length nn = 10)
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
    # Here we use 500 steps in each direction.
    omega3_min = 0.0
    omega3_max = 5.0
    omega3_step = 5.0 / 500.0

    omega0_min = 0.0
    omega0_max = 5.0
    omega0_step = 5.0 / 500.0

    omega0_values = np.arange(omega0_min, omega0_max + 1e-6, omega0_step)
    omega3_values = np.arange(omega3_min, omega3_max + 1e-6, omega3_step)

    # -------------------------------
    # Create a list of tasks for each (omega0, omega3) pair.
    # -------------------------------
    tasks = []
    for omega0 in omega0_values:
        for omega3 in omega3_values:
            tasks.append((omega0, omega3))

    # -------------------------------
    # Use multiprocessing Pool with tqdm to compute the population for each task.
    # -------------------------------
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)))
    
    # Reshape results into a 2D grid
    P3 = np.array(results).reshape(len(omega0_values), len(omega3_values))

    # -------------------------------
    # Plot a contour map of P3.
    # -------------------------------
    Omega3_grid, Omega0_grid = np.meshgrid(omega3_values, omega0_values)
    plt.contourf(Omega3_grid, Omega0_grid, P3, levels=50, cmap='viridis')
    plt.xlabel(r'$\Omega_3$')
    plt.ylabel(r'$\Omega_0$')
    plt.title(r'$P_3 = |E_3(z_{max})|^2$')
    plt.colorbar(label=r'$P_3$')
    plt.savefig('population_transfer_3state_500x500.png')
    plt.show()

    # Optionally, export the data to a file:
    # np.savetxt("c:\\data\\P3.dat", P3)
