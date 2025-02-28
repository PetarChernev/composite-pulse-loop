# Generate a grid of initial values for each phase
import itertools
import os
import numpy as np


def generate_cartesian_grid(step_size, num_phases):
    """Generate a Cartesian product of initial values for each phase."""
    values = np.arange(0, 2 * np.pi, step_size)  # Values in [0, 2π) with step `step_size`
    return list(itertools.product(values, repeat=num_phases))



def get_pickle_filename(name, N):
    return f"{name}_{N}_derivs.pkl"