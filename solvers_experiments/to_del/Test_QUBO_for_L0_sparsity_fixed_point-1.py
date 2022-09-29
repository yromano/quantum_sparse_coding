"""
Test the validity of the QUBO matrix that determines the L0 sparsity of
a vector x_ip of spins representing a vector of fixed point numbers x_i:
x_i = d * sum_p=1^P x_ip * (2 ** (p-1))

L0 sparsity = sum_i=1^N(x_i != 0)

Note: The general form of fixed point vector is:
x_i = x_min_i + d_i * sum_p=1^P x_ip * (2 ** (p-1))
For brevity we set x_min_i = 0 for all i, and d_i = d for all i.
In fact, the factor d is immaterial for the current script.

Harel Primack, 22.03.2022
"""

import random
import numpy as np
import qubo_matrices as QM
import itertools

# %% Set parameters

# Set a seed for RNG applications
seed = None

# Set the number of variables
N = 7
# Set the number of non-zero variables (cardinality)
K = 4
# Set the number of bits per variables
P = 5

# Weight for the hard conditions (F-functions)
w = 100

# %% Display info

print("N =", N, ",  K =", K, ",  P = ", P)

# %% Init the random engine

random.seed(a=seed)

# %% Select the non-zero variables

print("Selecting the non-zero entries")

inds_non_zero = random.sample(range(N), k=K)

print(" inds non-zero = ", inds_non_zero)

# %% Generate the bits for the variables

# Store the actual (real) solution
x_real  = np.zeros((N))
# Store the bit representation of the solution
num_spins_bits = N * P
x_bits = np.zeros(num_spins_bits)

# Go over the non-zero vars and randomize the bits
for ind_var in inds_non_zero:
    # The current var is not zero, randomize P bits, and make sure they are
    # not all zero
    while 1:
        bits = random.choices([0, 1], k=P)
        if np.all(np.array(bits) == 0):
            pass
        else:
            break
    # Store the randomized bits within the solution bits variable
    ind_spin_1 = P * ind_var
    ind_spin_2 = ind_spin_1 + P
    x_bits[ind_spin_1: ind_spin_2] = bits
    # Compute the actual value of the variable
    s = 0
    for ind in range(P):
        s += bits[ind] * (2 ** ind)
    x_real[ind_var] = s
    #print(" x[", ind_var, "] =", bits, ' = ', s)
print("x_real =", x_real)
print("x_bits =", x_bits)

# %% Generate the QUBO matrix to compute the cardinality
#    Note that the matrix includes also the ancilla spins

print("Computing QUBO matrix for L0 cardinality fixed-point")

Q, h = QM.generate_QUBO_L0_sparsity_fixed_point(0, N, P, w)

# %% Go over the ancilla spins with exhaustive search, and compute the
#    cardinality using QUBO

print("Applying exhaustive search over ancilla spins")

if P <= 2:
    num_ancilla_spins = 0
else:
    num_ancilla_spins = N * (P - 2)
print(" The total number of ancilla spins =", num_ancilla_spins)

# Generate all the combinations of the ancilla spins
ancilla_spins_combinations = itertools.product([0, 1], repeat=num_ancilla_spins)

# Go over all combinations of ancilla spins, and compute the minimum over Q
s = np.Inf
for ind, spins_ancilla_candidate in enumerate(ancilla_spins_combinations):
    spins_candidate = np.concatenate((x_bits, np.array(spins_ancilla_candidate)))
    tmp = (spins_candidate @ Q @ spins_candidate) + h
    if tmp < s:
        s = tmp
        spins_solution = spins_candidate

# %% Validate the QUBO solution with respect to the ground truth = K

print(" ")

if K == s:
    print("v Min energy of solution =", s, ",  K =", K, ' OK')
else:
    print("x Min energy of solution =", s, ",  K =", K, ' wrong')

print("Min solution (including ancilla) =", spins_solution)
