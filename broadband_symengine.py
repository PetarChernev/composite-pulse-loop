import random
import symengine as se
import sympy as sp
import pickle
import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import *
import signal
import functools

# Step 1: Define symbolic variables and parameters
N = 7  # Number of pulses (can be changed dynamically)
name = 'broadband_se'
phases = [se.Symbol(f'phi{i}') for i in range(1, N + 1)]  # Define phi1, phi2, ..., phiN

# Define the single-pulse propagator
def single_propagator(A, phi):
    cos = se.cos(A / 2)
    sin = se.sin(A / 2)
    return se.Matrix([
        [cos, -1j * sin * se.exp(-1j * phi)],
        [-1j * sin * se.exp(1j * phi), cos]
    ])

# Composite propagator for arbitrary N with anagram condition
def composite_propagator(A, phases):
    propagators = []
    for i in range(N):
        if i == 0 or i == N - 1:  # First and last pulses
            propagators.append(single_propagator(A, 0))
        else:  # Middle pulses follow the anagram condition
            phase_index = i if i <= N // 2 else (N - i - 1)
            propagators.append(single_propagator(A, phases[phase_index]))
    # Compute the product of propagators
    U_N = se.eye(2)
    for U in reversed(propagators):
        U_N = U @ U_N
        print(f"Simplifying propagator step")
        U_N[0, 0] = U_N[0, 0].simplify()
    return U_N

# Step 2: Compute the composite propagator
A = se.Symbol('A')  # Pulse area
U_N = composite_propagator(A, phases)
print('Simplifying propagator element.')
U_11 = U_N[0, 0]  # Top-left element of the propagator

# Sequential derivative computation
def compute_odd_derivatives_sequential(U_11, max_order):
    """Compute odd-order derivatives sequentially."""
    derivatives = []
    current_derivative = se.diff(U_11, A).simplify() # First-order derivative
    print("Computed first derivative")
    derivatives.append(current_derivative)

    for k in range(3, max_order + 1, 2):  # Compute 3rd, 5th, ..., (max_order)-th derivatives
        current_derivative = se.diff(current_derivative, A, 2).simplify()
        print(f"Computed derivative {k}.")
        derivatives.append(current_derivative)
    
    return derivatives

# Compute derivatives
derivs_cache = get_pickle_filename(name, N)
try:
    print(f"Loading derivatives from {derivs_cache}")
    with open(derivs_cache, 'rb') as file:
        derivatives = pickle.load(file)
except FileNotFoundError:
    print("\nComputing derivatives sequentially...")
    max_order = N - 2  # Compute up to (N-2)-th odd derivative
    derivatives = compute_odd_derivatives_sequential(U_11, max_order)
    
    # Substitute A = pi after all derivatives are computed
    sp_derivatives = []
    for i, derivative in enumerate(derivatives):
        print(f'Simplifying derivative {2 * i + 1}')
        sp_derivatives.append(sp.sympify(derivative).subs(A, se.pi).simplify())
    derivatives = sp_derivatives
    # derivatives = [sp.sympify(derivative) for derivative in derivatives]
    
    with open(derivs_cache, 'wb+') as file:
        pickle.dump(derivatives, file)

# Display the equations
# print("Equations to solve:")
# for i, derivative in enumerate(derivatives, start=1):
#     print(f"Derivative order {2 * i - 1}:", derivative)

# Generate a grid of initial values for each phase
step_size = np.pi / N  # Step size for grid search
phases_to_solve = N // 2  # Number of middle phases to solve for

def generate_cartesian_grid(step_size, num_phases):
    """Generate a Cartesian product of initial values for each phase."""
    values = np.arange(0, 2 * np.pi, step_size)  # Values in [0, 2π) with step `step_size`
    return list(product(values, repeat=num_phases))

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Timed out!")


def run_nsolve_with_timeout(expr, args, guess, timeout_sec=60):
    # Register signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)

    try:
        sol = sp.nsolve(expr, args, guess)
    finally:
        # Disable the alarm and restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return sol


# Worker function for nsolve
def solve_for_guess(initial_guess):
    # sp.nsolve can return a list or tuple of solutions, or a single solution
    # depending on how you structure your equations. Assume it's a list here.
    try:
        phi_sol_sympy = sp.nsolve(derivatives, phases[1:phases_to_solve + 1], initial_guess)
    except:
        return
    
    # Convert to real mod 2*pi
    try:
        phi_solution = []
        for phi in phi_sol_sympy:
            phi_e = phi.evalf()  # Convert symbolic expression to a numerical approximation
            # Extract real and imaginary parts
            re_val, im_val = phi_e.as_real_imag()
            
            # Check if the imaginary part is negligible
            if abs(im_val) < 1e-10:
                # Take the real part and do modulus
                phi_mod = float(re_val % (2 * np.pi))
                phi_solution.append(phi_mod)
            else:
                # If you truly want real solutions, discard this or return None
                return None
        return phi_solution
    except:
        return None

# Filter unique solutions
def filter_unique_solutions(solutions, tolerance=1e-2):
    """Filter unique solutions within a given tolerance."""
    unique_solutions = []
    for solution in solutions:
        if not any(np.allclose(solution, unique, atol=tolerance) for unique in unique_solutions):
            unique_solutions.append(solution)
    return unique_solutions

# Perform the grid search
if __name__ == "__main__":
    grid = generate_cartesian_grid(step_size, phases_to_solve)
    grid = random.sample(grid, int(0.1 * len(grid)))
    # grid = [(6*np.pi / N, 4*np.pi / N, 8*np.pi / N)]
    print(f"Total grid size: {len(grid)}")

    # Use multiprocessing with a progress bar
    with Pool(processes=12) as pool:
        results = list(tqdm(pool.imap_unordered(solve_for_guess, grid), total=len(grid), desc="Processing"))
    # results = list(tqdm(solve_for_guess(g) for g in grid))
    
    # Filter valid solutions and remove duplicates
    valid_solutions = [res for res in results if res is not None]
    unique_solutions = filter_unique_solutions(valid_solutions)

    # Display the results
    if unique_solutions:
        print(f"\nFound {len(unique_solutions)} unique solutions:")
        for solution in unique_solutions:
            print(f"Solution: {[phi / (np.pi / N) for phi in solution]}")
    else:
        print("\nNo solutions found.")
