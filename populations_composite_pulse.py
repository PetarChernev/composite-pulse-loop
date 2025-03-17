import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Patch


def get_pulse_phases(total_N):
    if total_N == 1:
        return [np.pi/4]
    if total_N % 2:
        raise ValueError("total_N must be even.")
    composite_half_phase = 5 * np.pi / 4
    if total_N == 2:
        phi = [0]
    elif total_N == 6:
        phi = [0, np.pi / 2, 0]
    elif total_N == 10:
        phi = [0, 5*np.pi/6, np.pi/3, 5*np.pi/6, 0]
    elif total_N == 14:
        phi = [p * np.pi / 12 for p in [0, 11, 10, 17, 10, 11, 0]]
    else:
        raise ValueError(f"Composite pulse not implemented for total_N = {total_N}.")
    return phi + [p + composite_half_phase for p in phi]

def get_pulse_breaks(total_N, max_t = 1.0):
    return np.arange(0, total_N + 1) * max_t / total_N


def omega_coeffs(z, omega0, phi, z_breaks):
    idx = np.searchsorted(z_breaks, z, side='right') - 1
    if idx < 0 or idx >= len(phi):
        return 0.0, 0.0
    return omega0*np.sin(phi[idx]), omega0*np.cos(phi[idx])

def ode_system(z, y, omega0, omega3, z_breaks, phi):
    E1, E2, E3 = y
    omega1, omega2 = omega_coeffs(z, omega0, phi, z_breaks)
    delta = 0.0
    dE1dz = -1j*omega1*E2 - omega3*E3
    dE2dz = -1j*np.conjugate(omega1)*E1 - 1j*delta*E2 - 1j*np.conjugate(omega2)*E3
    dE3dz = omega3*E1 - 1j*omega2*E2
    return [dE1dz, dE2dz, dE3dz]

def solve_system(total_N, omega0, omega3, resolution=0.001, max_t = 1):
    phi = get_pulse_phases(total_N)
    z_breaks = get_pulse_breaks(total_N, max_t)
    z_points = np.arange(0, max_t+ resolution, resolution)
    
    omega0 = omega0 * total_N / max_t
    omega3 = omega3 * total_N / max_t

  
    y0 = [1+0j, 0+0j, 0+0j]
    return solve_ivp(
        lambda z, y: ode_system(z, y, omega0, omega3, z_breaks, phi),
        t_span=(0, max_t),
        y0=y0,
        t_eval=z_points,
        method='RK45', atol=1e-8, rtol=1e-6
    )


if __name__ == "__main__":
    # Solve the system
    max_t = 1
    omega0 = 3
    omega3 =  1.3
    N = 6
    sol = solve_system(N, omega0, omega3, max_t=max_t)
    
    P1 = np.abs(sol.y[0])**2
    P2 = np.abs(sol.y[1])**2
    P3 = np.abs(sol.y[2])**2

    # Prepare stacked data
    cumP1 = P1
    cumP2 = P1 + P2
    cumP3 = P1 + P2 + P3  # should be ~1

    # --- Create a figure with 2 subplots (top: populations, bottom: pulses) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # -------------------- TOP SUBPLOT: Stacked population plot --------------------
    ax1.fill_between(sol.t, 0, cumP1,
                    facecolor='0.9', hatch='///', edgecolor='k', linewidth=0.5)
    ax1.fill_between(sol.t, cumP1, cumP2,
                    facecolor='0.7', hatch='\\\\', edgecolor='k', linewidth=0.5)
    ax1.fill_between(sol.t, cumP2, cumP3,
                    facecolor='0.5', hatch='xxx', edgecolor='k', linewidth=0.5)

    ax1.set_xlim(0, max_t)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Population')
    ax1.set_title(f'Population Dynamics for {N} Pulse{"s" if N > 1 else ""}')
    ax1.grid(True)

    # -------------------- BOTTOM SUBPLOT: Pulses from omega_coeffs ----------------
    # We'll sample the entire range [zmin, zmax], compute (omega1, omega2),
    # then compute amplitude = sqrt(omega1^2 + omega2^2) and phase = arctan2(omega1, omega2).
    z_fine = np.linspace(0, 1, 500)
    omega_vals = np.array([omega_coeffs(z, omega0, get_pulse_phases(N), get_pulse_breaks(N)) for z in z_fine])
    omega1_arr, omega2_arr = omega_vals[:,0], omega_vals[:,1]

    amplitude_arr = np.sqrt(omega1_arr**2 + omega2_arr**2)
    phase_arr = np.arctan2(omega1_arr, omega2_arr)  # in range [-pi, +pi]

    # Plot phase vs. z, coloring by amplitude in grayscale.
    # A simple way is to do a scatter plot:
    sc = ax2.scatter(z_fine, phase_arr, c=-amplitude_arr, cmap='gray',
                    edgecolor='none', s=5)

    ax2.set_xlim(0, 1)
    # Adjust your phase limits as desired
    ax2.set_ylim(-np.pi, np.pi)
    
    # Choose which tick locations you want
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]

    # Create LaTeX-style labels
    tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']

    # Assign them to the axis
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.set_xlabel('Propagator Coordinate z')
    ax2.set_ylabel('Phase')
    ax2.set_title('Pulse Phase')

    # -------------------- Legend Outside the Figure --------------------
    # We create custom legend patches in the order P3 (top), P2 (middle), P1 (bottom).
    legend_handles = [
        Patch(facecolor='0.5', hatch='xxx', edgecolor='k', label='P3'),
        Patch(facecolor='0.7', hatch='\\\\', edgecolor='k', label='P2'),
        Patch(facecolor='0.9', hatch='///', edgecolor='k', label='P1')
    ]

    # Tighten layout so subplots don't overlap
    plt.tight_layout()

    # Make some room on the right for the legend
    fig.subplots_adjust(right=0.9)

    # Place a single legend for the whole figure, anchored outside on the right.
    fig.legend(
        handles=legend_handles,
        loc='center right',       # vertically center on the right edge
        bbox_to_anchor=(0.98, 0.6),
        borderaxespad=0,
        labelspacing=3.0          # spread out entries vertically
    )

    plt.show()
