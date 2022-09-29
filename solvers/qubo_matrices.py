"""
Here we put functions to generate the QUBO matrices that are related to the
sparse regularization of linear systems.

We have the following options:
1. Binary or fixed-point (FP) variables
2. L0 or L1 regularization
3. QUBO matrices for the linear system, the sparsity condition and combined.

Harel Primack, 25.07.2021
25.07.2021 v1: Created.
26.04.2022 v2: Improved the L0 condition for fixed-point variables, reducing
the
"""

import numpy as np


# %% Generate QUBO matrix representing a linear system with binary solution
#    Ax = b, x_i = (0, 1)
#    The QUBO matrix represents the Hamiltonian
#    H = (Ax - b)_transposed * (Ax - b).

def generate_QUBO_linear_system_binary(A, b):
    # Get the size of the input vars
    num_measurements, num_spins = A.shape

    # Allocate and reset the output vars
    Q = np.zeros((num_spins, num_spins))
    h = 0

    # Fill in the AA term
    for ind_spin_1 in range(num_spins):
        for ind_spin_2 in range(num_spins):
            s = 0
            for i in range(num_measurements):
                s += A[i, ind_spin_1] * A[i, ind_spin_2]
            Q[ind_spin_1, ind_spin_2] = s

    # Fill in the Ab term
    for ind_spin in range(num_spins):
        s = 0
        for i in range(num_measurements):
            s += (-2.0) * A[i, ind_spin] * b[i]
        Q[ind_spin, ind_spin] += s

    # Set the offset = bb term
    h = b @ b

    return Q, h


# %% Generate QUBO matrix representing a linear system with fixed-point solution
#    Ax = b, x_i = x_min + d * sum_p=1_P x_ip * 2^(p-1)
#    we assume the same bias x_min and the sema dynamical range for all x_i.
#    The QUBO matrix represents the Hamiltonian
#    H = (Ax - b)_transposed * (Ax - b).

def generate_QUBO_linear_system_fixed_point(A, b, x_min, d, P):
    # Get the size of the input vars
    M, N = A.shape
    # The number of spins
    nn = N * P

    # Compute the QUBO matrix related to the binary case, which is used to
    # construct the QUBO matrix for the fixed-point case
    # Allocate and reset needed terms
    Q_LEb_1 = np.zeros((N, N))
    Q_LEb_2 = np.zeros((N, N))
    h_LEb = 0

    # Fill in the term Q_LEb_1
    for ind_spin_1 in range(N):
        for ind_spin_2 in range(N):
            s = 0
            for i in range(M):
                s += A[i, ind_spin_1] * A[i, ind_spin_2]
            Q_LEb_1[ind_spin_1, ind_spin_2] = s

    # Fill in the term Q_LEb_2
    for ind_spin in range(N):
        s = 0
        for i in range(M):
            s += (-2.0) * A[i, ind_spin] * b[i]
        Q_LEb_2[ind_spin, ind_spin] += s

    # Set the offset = bb term
    h_LEb = b @ b

    # Work out the full QUBO matrix
    Q_LEfp_1 = np.zeros((nn, nn))
    Q_LEfp_2 = np.zeros((nn, nn))
    h_LEfp = 0

    for ind_spin_1 in range(N):
        for ind_bit_1 in range(P):
            for ind_spin_2 in range(N):
                for ind_bit_2 in range(P):
                    ind_1 = ind_bit_1 + P * ind_spin_1
                    ind_2 = ind_bit_2 + P * ind_spin_2
                    Q_LEfp_1[ind_1, ind_2] = \
                        2 ** (ind_bit_1 + ind_bit_2) * \
                        Q_LEb_1[ind_spin_1, ind_spin_2] * (d ** 2)

    for ind_spin in range(N):
        for ind_bit in range(P):
            ind = ind_bit + P * ind_spin
            s = 0
            for ind_1 in range(N):
                s += Q_LEb_1[ind_spin, ind_1]
            Q_LEfp_2[ind, ind] = \
                (2 ** ind_bit) * d * \
                (Q_LEb_2[ind_spin, ind_spin] + 2 * x_min * s)

    Q_LEfp = Q_LEfp_1 + Q_LEfp_2

    s1 = 0
    for ind_spin_1 in range(N):
        for ind_spin_2 in range(N):
            s1 += Q_LEb_1[ind_spin_1, ind_spin_2]
    s2 = 0
    for ind in range(N):
        s2 += Q_LEfp_2[ind, ind]
    h_LEfp = s1 * (x_min ** 2) + s2 * x_min + h_LEb

    return Q_LEfp, h_LEfp


# %% Generate QUBO matrix for the condition of L0 sparsity of BINARY variables
#    It is the same matrix for the L1 norm for binary variables

def generate_QUBO_L0_sparsity_binary(num_spins):
    # Allocate and reset the matrix and the offset
    Q = np.zeros((num_spins, num_spins))
    h = 0.

    # The sum of the spins is the cardinality, which is to be minimized
    for ind_spin in range(num_spins):
        Q[ind_spin, ind_spin] = 1.

    return Q, h


# %% Generate QUBO matrix for the condition of L1 sparsity of BINARY variables
#    It is the same matrix for the L0 norm for binary variables
def generate_QUBO_L1_sparsity_binary(num_spins):
    return generate_QUBO_L0_sparsity_binary(num_spins)


#    For the time being, and for brevity, we restrict ourselves to the case
#    of x_min = 0, meaning that x_i = 0 iff all the spins related to it x_ip
#    are zero; in the more general case this needs to be modified.
#
#    We construct the QUBO matrix to include the bit spins of x_i (1-bit and 2-bit)
def generate_QUBO_L0_sparsity_fixed_point(
        x_min, N, num_bits_per_var, w):
    # Apply validity check
    if x_min != 0:
        print("Error: Currently, the case with x_min ~= 0 is not implemented")
        1 / 0

    # Treat the binary case separately, since in the sequel we assume 2 or
    # more bits per variable
    if num_bits_per_var == 1:
        #print(" Applying the binary L0 sparsity matrix")
        return generate_QUBO_L0_sparsity_binary(N)

    # Treat the 2-bit per var case separately, since it requires no ancilla
    # variables
    if num_bits_per_var == 2:
        num_spins = N * num_bits_per_var
        # Allocate and reset the output matrices
        Q = np.zeros((num_spins, num_spins))
        h = 0
        # Loop over all variables
        for ind_var in range(N):
            # Compute the indices of the 2 spins related to the current var
            i0 = ind_var * num_bits_per_var
            # p = 1
            ind_spin_1 = i0
            # p = 2
            ind_spin_2 = i0 + 1
            # Set the entries for the expression x_i1 + x_i2 - x_i1 * x_i2
            Q[ind_spin_1, ind_spin_1] = 1.
            Q[ind_spin_2, ind_spin_2] = 1.
            Q[ind_spin_1, ind_spin_2] = (-1. / 2.)
            Q[ind_spin_2, ind_spin_1] = (-1. / 2.)

        return Q, h
    if num_bits_per_var >= 3:
        print("Error: Currently, the case with num_bits_per_var >= 3 is not implemented")
        1 / 0


# %% Generate QUBO matrix for a linear system with binary variables +
#    L0 sparsity condition

def generate_QUBO_linear_system_L0_sparsity_binary(A, b, lam):
    # Generate QUBO matrix representing a general linear system
    Q1, h1 = generate_QUBO_linear_system_binary(A, b)

    # Generate QUBO matrix for the condition of L0 sparsity of BINARY variables
    M, N = A.shape
    Q2, h2 = generate_QUBO_L0_sparsity_binary(N)

    # Compose the total QUBO matrix
    Q = Q1 + lam * Q2
    h = h1 + lam * h2

    # return Q, h, Q1, h1, Q2, h2
    return Q, h


# %% Generate QUBO matrix for a linear system with fixed-point variables +
#    L0 sparsity condition
def generate_QUBO_linear_system_L0_sparsity_fixed_point(
        A, b, x_min, d, P, lam, w):
    # Generate QUBO matrix representing a general linear system
    Q1, h1 = generate_QUBO_linear_system_fixed_point(A, b, x_min, d, P)

    # Generate QUBO matrix for the condition of L0 sparsity of fixed point
    # variables
    M, N = A.shape
    Q2, h2 = generate_QUBO_L0_sparsity_fixed_point(x_min, N, P, w)

    # Compose the total QUBO matrix
    n = Q1.shape[0]
    Q = np.zeros_like(Q2)
    Q[0:n, 0:n] = Q1
    Q += lam * Q2
    h = h1 + lam * h2

    # return Q, h, Q1, h1, Q2, h2
    return Q, h
