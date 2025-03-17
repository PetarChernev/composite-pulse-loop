import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

from populations_composite_pulse import get_pulse_breaks, get_pulse_phases, ode_system


def population(omega0, omega3, zp_cumsum, phi, zmin, zmax, z_points):
    """
    Solves the ODE system from zmin to zmax with initial conditions
      E1(zmin)=1, E2(zmin)=0, E3(zmin)=0
    and returns |E3(zmax)|^2.
    """
    y0 = [1+0j, 0+0j, 0+0j]
    sol = solve_ivp(lambda z, y: ode_system(z, y, omega0, omega3, zp_cumsum, phi),
                    t_span=(zmin, zmax), y0=y0, method='DOP853', t_eval=z_points,
                    atol=1e-8, rtol=1e-6)
    E3_final = sol.y[2, -1]
    return np.abs(E3_final)**2

# Worker function defined at the top level for pickling.
def worker(args):
    omega0, omega3 = args
    omega0 = omega0 * total_N / max_t
    omega3 = omega3 * total_N / max_t
    return population(omega0, omega3, z_breaks, phi, 0, max_t, z_points)

if __name__ == '__main__':
    total_N = 14
    max_t = 1
    resolution = 0.01
    phi = get_pulse_phases(total_N)
    z_breaks = get_pulse_breaks(total_N, max_t)
    z_points = np.arange(0, max_t+ resolution, resolution)
    
    omega3_min = 0.0
    omega3_max = 3.0
    omega3_step = 3.0 / 50.0

    omega0_min = 0.0
    omega0_max = 5.0
    omega0_step = 5.0 / 50.0

    omega0_values = np.arange(omega0_min, omega0_max + 1e-6, omega0_step)
    omega3_values = np.arange(omega3_min, omega3_max + 1e-6, omega3_step)
    
    tasks = []
    for omega0 in omega0_values:
        for omega3 in omega3_values:
            tasks.append((omega0, omega3))

    with Pool() as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))


    P3 = np.array(results).reshape(len(omega0_values), len(omega3_values))

    Omega3_grid, Omega0_grid = np.meshgrid(omega3_values, omega0_values)
    plt.figure(figsize=(4, 6))
    plt.contourf(Omega3_grid, Omega0_grid, P3, levels=50, cmap='gray')
    plt.xlabel(r'$\Omega_3$')
    plt.ylabel(r'$\Omega_0$')
    plt.title(r'$P_3 = |E_3(t_{max})|^2$' + f", {total_N} pulses")
    plt.colorbar(label=r'$P_3$')
    plt.savefig('population_transfer_3state_500x500.png')
    plt.show()