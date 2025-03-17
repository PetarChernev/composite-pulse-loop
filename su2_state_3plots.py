from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# Import your ODE system from populations_composite_pulse.py
# from populations_composite_pulse import ode_system

def ode_system(z, y, omega0, omega3, zp_cumsum, phi):
    """
    Dummy placeholder for your actual ODE system.
    Replace this with the real 'ode_system' from populations_composite_pulse.py.
    """
    # y is [E1, E2, E3].
    # This function must return dE1/dz, dE2/dz, dE3/dz.
    # For now, just return zeros to make this script standalone.
    return [0.0j, 0.0j, 0.0j]

def Population(omega0, omega3, zp_cumsum, phi, zmin, zmax, z_points):
    """
    Solves the ODE system from zmin to zmax with initial conditions
      E1(zmin)=1, E2(zmin)=0, E3(zmin)=0
    and returns |E3(zmax)|^2.
    """
    y0 = [1+0j, 0+0j, 0+0j]
    sol = solve_ivp(
        lambda z, y: ode_system(z, y, omega0, omega3, zp_cumsum, phi),
        t_span=(zmin, zmax),
        y0=y0,
        method='DOP853',
        t_eval=z_points,
        atol=1e-8,
        rtol=1e-6
    )
    E3_final = sol.y[2, -1]
    return np.abs(E3_final)**2

def worker(args):
    """
    Worker function for parallel processing.
    """
    omega0, omega3, zp_cumsum, phi, zmin, zmax, z_points = args
    return Population(omega0, omega3, zp_cumsum, phi, zmin, zmax, z_points)

def compute_population_grid(phi, nn, Eta, zstep,
                            omega0_values, omega3_values,
                            n_processes=None):
    """
    Given a sequence of phases 'phi' of length nn,
    compute the 2D grid of final populations P3 over
    the specified range of omega0_values and omega3_values.
    """

    # Build zp = [0, Eta, Eta, ..., Eta] of length nn+1
    zp = [0] + [Eta]*nn
    zp_cumsum = np.array(np.cumsum(zp))

    # z range
    zmin = 0.0
    zmax = nn * Eta
    z_points = np.arange(zmin, zmax + zstep, zstep)

    # Prepare tasks
    tasks = []
    for omega0 in omega0_values:
        for omega3 in omega3_values:
            tasks.append((omega0, omega3, zp_cumsum, phi, zmin, zmax, z_points))

    # Run either in parallel or single-threaded
    if n_processes is None or n_processes > 1:
        with Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))
    else:
        # Single process for debugging
        results = [worker(t) for t in tqdm(tasks)]

    # Reshape results into a 2D array
    P3 = np.array(results).reshape(len(omega0_values), len(omega3_values))
    return P3

if __name__ == '__main__':

    # Common scan ranges for Omega0 and Omega3
    omega3_min, omega3_max = 0.0, 3.0
    omega3_step = 3.0 / 100.0  # ~0.03 steps
    omega0_min, omega0_max = 0.0, 5.0
    omega0_step = 5.0 / 100.0  # ~0.05 steps

    omega3_values = np.arange(omega3_min, omega3_max + 1e-9, omega3_step)
    omega0_values = np.arange(omega0_min, omega0_max + 1e-9, omega0_step)

    # We'll create the corresponding meshgrid for plotting
    Omega3_grid, Omega0_grid = np.meshgrid(omega3_values, omega0_values)

    # We will use the same z-step for all computations
    zstep = 0.01
    Eta = 1.0

    # --------------------------------------------------------------------
    # 1) Single pulse (nn = 1, phi = [0])
    # --------------------------------------------------------------------
    phi_1 = [0]
    nn_1 = 1
    P3_1 = compute_population_grid(phi=phi_1,
                                   nn=nn_1,
                                   Eta=Eta,
                                   zstep=zstep,
                                   omega0_values=omega0_values,
                                   omega3_values=omega3_values,
                                   n_processes=None)  # or set n_processes to e.g. 4

    # --------------------------------------------------------------------
    # 2) Six pulses (nn = 6, phases = (0, pi/2, 0, 5pi/4, 7pi/4, 5pi/4))
    # --------------------------------------------------------------------
    phi_6 = [0, np.pi/2, 0, 5*np.pi/4, 7*np.pi/4, 5*np.pi/4]
    nn_6 = 6
    P3_6 = compute_population_grid(phi=phi_6,
                                   nn=nn_6,
                                   Eta=Eta,
                                   zstep=zstep,
                                   omega0_values=omega0_values,
                                   omega3_values=omega3_values,
                                   n_processes=None)

    # --------------------------------------------------------------------
    # 3) Ten pulses (the original sequence you had)
    # --------------------------------------------------------------------
    nn_10 = 10
    # "Alpha" from your original code:
    Alpha = np.pi / 2
    # The 10 phases from your script
    phi_10 = [
        0,
        5*np.pi/6,
        np.pi/3,
        5*np.pi/6,
        0,
        np.pi + Alpha/2,
        11*np.pi/6 + Alpha/2,
        4*np.pi/3 + Alpha/2,
        11*np.pi/6 + Alpha/2,
        np.pi + Alpha/2
    ]

    P3_10 = compute_population_grid(phi=phi_10,
                                    nn=nn_10,
                                    Eta=Eta,
                                    zstep=zstep,
                                    omega0_values=omega0_values,
                                    omega3_values=omega3_values,
                                    n_processes=None)

    # --------------------------------------------------------------------
    # Plot all three side by side in the same figure
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    # To keep a uniform color scale, find the global min/max of P3
    p3_min = 0.0
    p3_max = max(P3_1.max(), P3_6.max(), P3_10.max())

    # Single pulse
    c0 = axes[0].contourf(Omega3_grid, Omega0_grid, P3_1,
                          levels=50, cmap='gray',
                          vmin=p3_min, vmax=p3_max)
    axes[0].set_title('1 pulse')
    axes[0].set_xlabel(r'$\Omega_3$')
    axes[0].set_ylabel(r'$\Omega_0$')

    # Six pulses
    c1 = axes[1].contourf(Omega3_grid, Omega0_grid, P3_6,
                          levels=50, cmap='gray',
                          vmin=p3_min, vmax=p3_max)
    axes[1].set_title('6 pulses')
    axes[1].set_xlabel(r'$\Omega_3$')
    axes[1].set_ylabel(r'$\Omega_0$')

    # Ten pulses
    c2 = axes[2].contourf(Omega3_grid, Omega0_grid, P3_10,
                          levels=50, cmap='gray',
                          vmin=p3_min, vmax=p3_max)
    axes[2].set_title('10 pulses')
    axes[2].set_xlabel(r'$\Omega_3$')
    axes[2].set_ylabel(r'$\Omega_0$')

    # Put one colorbar on the right side
    cb = fig.colorbar(c2, ax=axes.ravel().tolist(), label=r'$|E_3(z_{\max})|^2$')

    plt.tight_layout()
    plt.savefig('population_transfer_single_6_10_pulses.png', dpi=300)
    plt.show()
