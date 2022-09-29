"""
In this file we put solvers for QUBO problems.

We look for a spin vector s, whose entries are taken from (0, 1),
that minimizes the energy function H = sQs + h.

We only include a function to effect an exhaustive search of the minimal
solutions.

Harel Primack, 13.03.2022

"""

import numpy as np
import itertools

# %% Apply an exhaustive search over the spin space related to a QUBO matrix,
#    and find the minimal solution.
#    This function is generic, and independent of the nature of the original
#    problem.

def solve_QUBO_exhaustive(Q, h, display=False):
    N = Q.shape[0]
    num_spins = N

    # Construct all the spin combinations
    spin_combinations = itertools.product([0, 1], repeat=N)
    num_spin_combinations = 2 ** N
    if display:
        print(" Number of spin combinations = ", num_spin_combinations)

    # Go over all the spin combinations, and compute the value of the
    # Hamiltonian, find the minimal one and its related solution
    spins_solution = []
    H_min          = np.Inf
    # Interval for printing progress
    dd = int(num_spin_combinations / 10)
    if dd < 100:
        dd = 100

    for ind, spins in enumerate(spin_combinations):
        if display & (ind % dd == 0):
            print(" Computed ", ind+1, " combinations")
        H = spins @ Q @ spins + h
        if H < H_min:
            spins_solution = spins
            H_min = H

    # Extract the solution
    H_QUBO_solution     = H_min
    spins_QUBO_solution = spins_solution

    return spins_QUBO_solution, H_QUBO_solution
