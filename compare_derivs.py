import symengine as se
import sympy as sp
import pickle
from utils import *

# Step 1: Define symbolic variables and parameters
N = 5  # Number of pulses (can be changed dynamically)
name_symengine = 'broadband_se'
name_sympy = 'broadband_sp'
phases_se = [se.Symbol(f'phi{i}') for i in range(1, N + 1)]  # Define phi1, phi2, ..., phiN (SymEngine)
phases_sp = [sp.Symbol(f'phi{i}') for i in range(1, N + 1)]  # Define phi1, phi2, ..., phiN (SymPy)

# Define the single-pulse propagator
def single_propagator(engine, A, phi):
    cos = engine.cos(A / 2)
    sin = engine.sin(A / 2)
    exp = engine.exp
    return engine.Matrix([
        [cos, -1j * sin * exp(-1j * phi)],
        [-1j * sin * exp(1j * phi), cos]
    ])

# Composite propagator for arbitrary N with anagram condition
def composite_propagator(engine, A, phases):
    propagators = []
    for i in range(N):
        if i == 0 or i == N - 1:  # First and last pulses
            propagators.append(single_propagator(engine, A, 0))
        else:  # Middle pulses follow the anagram condition
            phase_index = i if i <= N // 2 else (N - i - 1)
            propagators.append(single_propagator(engine, A, phases[phase_index]))
    # Compute the product of propagators
    U_N = engine.eye(2)
    for U in reversed(propagators):
        U_N = U @ U_N
    return U_N

# Sequential derivative computation
def compute_odd_derivatives_sequential(engine, U_11, A, max_order):
    """Compute odd-order derivatives sequentially."""
    derivatives = []
    current_derivative = engine.diff(U_11, A).simplify()
    derivatives.append(current_derivative)

    for k in range(3, max_order + 1, 2):  # Compute 3rd, 5th, ..., (max_order)-th derivatives
        current_derivative = engine.diff(current_derivative, A).simplify()
        derivatives.append(current_derivative)
    
    return derivatives

# Compute derivatives using SymEngine
A_se = se.Symbol('A')
U_N_se = composite_propagator(se, A_se, phases_se)
U_11_se = U_N_se[0, 0]

derivs_cache_se = get_pickle_filename(name_symengine, N)
try:
    with open(derivs_cache_se, 'rb') as file:
        derivatives_se = pickle.load(file)
except FileNotFoundError:
    print("\nComputing derivatives sequentially with SymEngine...")
    max_order = N - 2  # Compute up to (N-2)-th odd derivative
    derivatives_se = compute_odd_derivatives_sequential(se, U_11_se, A_se, max_order)
    
    # Substitute A = pi after all derivatives are computed
    derivatives_se = [derivative.subs(A_se, se.pi).simplify() for derivative in derivatives_se]
    
    with open(derivs_cache_se, 'wb+') as file:
        pickle.dump(derivatives_se, file)

# Compute derivatives using SymPy
A_sp = sp.Symbol('A')
U_N_sp = composite_propagator(sp, A_sp, phases_sp)
U_11_sp = U_N_sp[0, 0]

derivs_cache_sp = get_pickle_filename(name_sympy, N)
try:
    with open(derivs_cache_sp, 'rb') as file:
        derivatives_sp = pickle.load(file)
except FileNotFoundError:
    print("\nComputing derivatives sequentially with SymPy...")
    max_order = N - 2  # Compute up to (N-2)-th odd derivative
    derivatives_sp = compute_odd_derivatives_sequential(sp, U_11_sp, A_sp, max_order)
    
    # Substitute A = pi after all derivatives are computed
    derivatives_sp = [sp.simplify(derivative.subs(A_sp, sp.pi)) for derivative in derivatives_sp]
    
    with open(derivs_cache_sp, 'wb+') as file:
        pickle.dump(derivatives_sp, file)

# Display and compare the equations
print("\nSymEngine derivatives:")
for i, derivative in enumerate(derivatives_se, start=1):
    print(f"Derivative order {2 * i - 1} (SymEngine): {derivative}")

print("\nSymPy derivatives:")
for i, derivative in enumerate(derivatives_sp, start=1):
    print(f"Derivative order {2 * i - 1} (SymPy): {derivative}")

# Compare the results
print("\nComparison of derivatives (SymEngine vs SymPy):")
for i, (der_se, der_sp) in enumerate(zip(derivatives_se, derivatives_sp), start=1):
    print(f"Derivative order {2 * i - 1}:")
    print(f"  SymEngine: {der_se}")
    print(f"  SymPy:    {der_sp}")
    print(f"  Match:    {sp.simplify(der_se - sp.sympify(der_sp)) == 0}")
