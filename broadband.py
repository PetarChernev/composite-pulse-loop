from concurrent.futures import ProcessPoolExecutor
import random
import sympy as sp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from mpmath import mp
import pickle

from utils import *

# Step 1: Define symbolic variables and parameters
N = 5  # Number of pulses (can be changed dynamically)
name = "broadband"

# Dynamically define the phases
phases = sp.symbols(f'phi1:{N + 1}')  # Define phi1, phi2, ..., phiN

# Define the single-pulse propagator
def single_propagator(A, phi):
    cos = sp.cos(A / 2)
    sin = sp.sin(A / 2)
    return sp.Matrix([
        [cos, -1j * sin * sp.exp(-1j * phi)],
        [-1j * sin * sp.exp(1j * phi), cos]
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
    U_N = sp.eye(2)
    for U in reversed(propagators):
        U_N = U @ U_N
    return U_N

# Step 2: Compute the composite propagator
A = sp.symbols('A')  # Pulse area
U_N = composite_propagator(A, phases)
U_11 = U_N[0, 0]  # Top-left element of the propagator


# Parallelized function to compute a derivative
def compute_derivative(k):
    return sp.diff(U_11, A, k).subs(A, sp.pi).simplify()

derivs_cache = get_pickle_filename(name,N)
try:
    file = open(derivs_cache, 'rb')
except FileNotFoundError:
    # Step 3: Derive the derivatives for the broadband condition (parallelized)
    print("\nComputing derivatives in parallel...")
    odd_orders = range(1, N - 1, 2)  # Odd-order derivatives
    with ProcessPoolExecutor() as executor:
        # Use tqdm to wrap the parallel computation with a progress bar
        derivatives = list(tqdm(executor.map(compute_derivative, odd_orders), total=len(odd_orders)))
    with open(derivs_cache, 'wb+') as file:
        pickle.dump(derivatives, file)
else:
    derivatives = pickle.load(file)
    file.close()

# Display the equations
print("Equations to solve:")
for i, derivative in enumerate(derivatives, start=1):
    print(f"Derivative order {2 * i - 1}:", derivative)

step_size = np.pi / N  # Step size for grid search
phases_to_solve = N // 2  # Number of middle phases to solve for

# Set higher precision for numerical solving
mp.dps = 50  # Increase precision to 50 decimal places


# Rescale derivatives for better numerical stability
scaled_derivatives = [
    derivative / (max(abs(c) for c in derivative.as_coefficients_dict().values()) or 1)
    for derivative in derivatives
]

# Worker function for multiprocessing
def solve_for_guess(initial_guess):
    """Attempt to solve the system for a given initial guess."""
    try:
        phi_solution = sp.nsolve(scaled_derivatives, phases[1:phases_to_solve + 1], initial_guess)
        # Normalize solutions to [0, 2π]
        phi_solution = [float(sp.re(phi) % (2 * np.pi)) for phi in phi_solution]
        return phi_solution
    except Exception:
        return None

# Function to filter unique solutions
def filter_unique_solutions(solutions, tolerance=1e-1):
    """Filter unique solutions within a given tolerance."""
    unique_solutions = []
    for solution in solutions:
        if not any(np.allclose(solution, unique, atol=tolerance) for unique in unique_solutions):
            unique_solutions.append(solution)
    return unique_solutions

# Perform the grid search using multiprocessing
if __name__ == "__main__":
    grid = generate_cartesian_grid(step_size, phases_to_solve)
    # grid = random.sample(grid, k=int(0.02 * len(grid)))
    print(f"Total grid size: {len(grid)}")
    
    # Use multiprocessing with a progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(solve_for_guess, grid), total=len(grid), desc="Processing"))
    
    # Filter valid solutions and remove duplicates
    valid_solutions = [res for res in results if res is not None]
    unique_solutions = filter_unique_solutions(valid_solutions)

    # Define a tolerance for checking if derivatives are close to zero
    tolerance = 1e-6  # Adjust as needed

    if unique_solutions:
        print(f"\nFound {len(unique_solutions)} unique solutions:")
        for solution in unique_solutions:
            # Evaluate the derivatives for the current solution
            derivative_values = [
                derivative.subs({phases[j + 1]: solution[j] for j in range(len(solution))}).evalf()
                for derivative in derivatives
            ]
            
            # Check if all derivative values are close to zero
            if all(abs(value) < tolerance for value in derivative_values):
                print(f"Solution: {[s / (np.pi / N) for s in solution]}")
                print(f"Derivative values for this solution: {derivative_values}")
    else:
        print("\nNo solutions found.")
