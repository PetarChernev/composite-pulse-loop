#!/usr/bin/env python3

import os
import pickle
import numpy as np
import sympy as sp
import symengine as se
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Simple Utilities (If you have these in another file, just import them.)
# ------------------------------------------------------------------------------

def composite_propagator(A, phases, N=9):
    """
    Build the composite propagator for a 2-level system with N pulses,
    each having the same pulse area A but different phases phi_i.
    Returns a 2x2 symengine.Matrix.
    """
    def single_propagator(A_sym, phi_sym):
        cos = se.cos(A_sym / 2)
        sin = se.sin(A_sym / 2)
        return se.Matrix([
            [cos, -1j * sin * se.exp(-1j * phi_sym)],
            [-1j * sin * se.exp(1j * phi_sym), cos]
        ])

    propagators = []
    for i in range(N):
        if i == 0 or i == N - 1:
            # First and last pulses have phase 0
            propagators.append(single_propagator(A, 0))
        else:
            # anagram condition
            phase_index = i if i <= N // 2 else (N - i - 1)
            propagators.append(single_propagator(A, phases[phase_index]))

    U_N = se.eye(2)
    for U_i in reversed(propagators):
        U_N = U_i @ U_N

    return U_N


def compute_odd_derivatives_sequential(expr, A_sym, max_order):
    """
    Compute 1st, 3rd, 5th, ..., up to max_order-th derivative wrt A_sym.
    Returns a list of those derivatives (symbolic expressions).
    """
    derivatives = []
    current_derivative = expr.diff(A_sym, 1).simplify()
    derivatives.append(current_derivative)

    # 3rd, 5th, 7th, ...
    for k in range(3, max_order + 1, 2):
        current_derivative = current_derivative.diff(A_sym, 2).simplify()
        derivatives.append(current_derivative)

    return derivatives


# ------------------------------------------------------------------------------
# 1) Function to load or compute U_N (and optionally its derivatives),
#    caching in a pickle file. Convert to/from Sympy for pickling.
# ------------------------------------------------------------------------------

def get_propagator_and_derivs(N=9, name='broadband_se', compute_derivatives=True):
    """
    Attempt to load the composite propagator U_N and (optionally) its odd-order derivatives
    from pickle files. If not found, compute them, save them, and return them.
    """
    # Symbolic variables
    A = se.Symbol('A', real=True, positive=True)
    phases = [se.Symbol(f'phi{i}') for i in range(1, N + 1)]

    # Unique filenames for the pickles
    propagator_filename = f"{name}_U_{N}.pkl"
    derivs_filename = f"{name}_derivs_{N}_odd.pkl"

    # --- Load or compute the propagator ---
    if os.path.isfile(propagator_filename):
        # Load from pickle (stored as Sympy Matrix), then convert to SymEngine
        with open(propagator_filename, 'rb') as f:
            U_N_sympy = pickle.load(f)
        print(f"[INFO] Loaded propagator from: {propagator_filename}")
        U_N = se.Matrix(U_N_sympy)
    else:
        print("[INFO] Propagator not found. Computing now...")
        U_N_engine = composite_propagator(A, phases, N)
        # Convert to Sympy for pickling
        U_N_sympy = sp.Matrix(U_N_engine)
        with open(propagator_filename, 'wb') as f:
            pickle.dump(U_N_sympy, f)
        print(f"[INFO] Saved propagator to: {propagator_filename}")
        U_N = U_N_engine

    # --- Load or compute the derivatives if needed ---
    derivatives = None
    if compute_derivatives:
        if os.path.isfile(derivs_filename):
            with open(derivs_filename, 'rb') as f:
                derivatives_sympy = pickle.load(f)
            print(f"[INFO] Loaded derivatives from: {derivs_filename}")
            # Convert each to SymEngine
            derivatives = [se.sympify(d) for d in derivatives_sympy]
        else:
            print("[INFO] Derivatives not found. Computing now...")
            U_11 = U_N[0, 0]
            max_order = N - 2
            derivatives_engine = compute_odd_derivatives_sequential(U_11, A, max_order)
            # Convert them to Sympy for pickling
            derivatives_sympy = [sp.sympify(d) for d in derivatives_engine]
            with open(derivs_filename, 'wb') as f:
                pickle.dump(derivatives_sympy, f)
            print(f"[INFO] Saved derivatives to: {derivs_filename}")
            derivatives = derivatives_engine

    return U_N, derivatives


# ------------------------------------------------------------------------------
# 2) Function to plot the transition probability for a SINGLE solution
#    as a function of A in [0, 2π].
# ------------------------------------------------------------------------------

def plot_transition_probability(U_N, solution, N=9, num_points=200):
    """
    Plot the transition probability P(0->1) = |U_{10}(A)|^2
    for a SINGLE solution of the middle phases.

    Parameters
    ----------
    U_N : symengine.Matrix
        Symbolic 2x2 matrix of the composite propagator, depending on A and phases.
    solution : list of float
        The phase values for phi2..phi(N-1).
    N : int
        Number of pulses.
    num_points : int
        Number of points in the interval [0, 2π] used for plotting.
    """
    # We'll find the symbolic variable for A:
    # Typically U_N has A as a free symbol, plus the phases.
    # If you prefer a direct approach:
    A_sym = se.Symbol('A', real=True, positive=True)

    # Build a substitution dict for the phases:
    phases = [se.Symbol(f'phi{i}') for i in range(1, N + 1)]
    subs_dict = {}
    # phi1 = phiN = 0
    subs_dict[phases[0]] = 0
    subs_dict[phases[-1]] = 0
    # Fill in the middle pulses from 'solution'
    for i in range(1, N - 1):
        subs_dict[phases[i]] = solution[i - 1]

    # Evaluate the transition amplitude U_{10} at different A in [0, 2π]
    A_vals = np.linspace(0, 2 * np.pi, num_points)
    P_vals = []
    for A_val in A_vals:
        val = U_N[1, 0].subs({**subs_dict, A_sym: A_val})
        val_cpx = complex(val.evalf())
        P_vals.append(abs(val_cpx)**2)  # Probability = |U_{10}|^2

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(A_vals, P_vals, 'b-', lw=2)
    plt.title("Transition Probability (|0> → |1>) vs. Pulse Area A")
    plt.xlabel("Pulse area A")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 3) Function to evaluate and print the odd derivatives of U_11 at A=pi
#    for a single solution.
# ------------------------------------------------------------------------------

def evaluate_derivatives_at_pi(derivatives, solution, N=9):
    """
    Given a list of odd-order derivatives of U_11 wrt A (symbolic expressions),
    evaluate them at A=pi and the given solution for the middle phases,
    then print the results.

    Parameters
    ----------
    derivatives : list of symengine expressions (1st, 3rd, 5th, ... derivative)
    solution : list of float
        The phase values for phi2..phi(N-1).
    N : int
        Number of pulses.
    """
    # We'll define A_sym for clarity:
    A_sym = se.Symbol('A', real=True, positive=True)
    phases = [se.Symbol(f'phi{i}') for i in range(1, N + 1)]

    # Build the substitution dictionary
    subs_dict = {}
    subs_dict[phases[0]] = 0
    subs_dict[phases[-1]] = 0
    for i in range(1, N - 1):
        subs_dict[phases[i]] = solution[i - 1]

    # Also substitute A = pi
    subs_dict[A_sym] = np.pi

    print("\nEvaluating odd derivatives of U_11 at A = π for the given solution:")
    print(f"Solution (phi2..phi{N-1}): {solution}")
    for d_idx, derivative_expr in enumerate(derivatives, start=1):
        derivative_order = 2 * d_idx - 1  # (1st, 3rd, 5th, ...)
        val = derivative_expr.subs(subs_dict)
        val_simpl = val.simplify()  # optionally
        print(f"  {derivative_order}-th derivative: {val_simpl.evalf()}")


# ------------------------------------------------------------------------------
# Example main() to show usage
# ------------------------------------------------------------------------------
def main():
    N = 7
    name = "broadband_se"

    # Load or compute the propagator and derivatives
    U_N, derivatives = get_propagator_and_derivs(N=N, name=name, compute_derivatives=True)

    # List of solutions with numbers rounded to integers where appropriate
    solutions = [
        [8.610374193138274, 8.61037419313828, 14],
        [5.389625806861726, 5.38962580686172, 0],
        [8, 10, 6],
        [6, 4, 8],
        [4, 12, 10],
        [10, 2, 4],
        [12, 8, 2],
        [2, 6, 12],
        [5.389625806871688, 5.3896258068522656, 14]
    ]

    for example_solution in solutions:
        # Adjust solution to radians
        example_solution = [s * np.pi / N for s in example_solution + example_solution[:2][::-1]]

        # Plot the transition probability for this solution
        plot_transition_probability(U_N, example_solution, N=N, num_points=300)

        # Evaluate derivatives at A=pi for this solution
        evaluate_derivatives_at_pi(derivatives, example_solution, N=N)


if __name__ == "__main__":
    main()
