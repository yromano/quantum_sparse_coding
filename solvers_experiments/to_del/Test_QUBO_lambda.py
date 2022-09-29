"""
Test the behaviour of the QUBO solution as a function of lambda for linear
systems with binary variables
"""


import sys

base_path = "/Users/romano/Google Drive/lightsolver/repo_sparse/"

sys.path.append(base_path + "sparse_coding/")
sys.path.append(base_path + "solvers/")
sys.path.append(base_path + "utils/")

import numpy as np
from solvers import annealer
import matplotlib.pyplot as plt
import random_linear_system as RLS

    
# %% Parameters

# The seed for the RNG
seed = None
#seed = 0

# Number of (features) variables
N = 16
# Number of non-zero variables (cardinality)
K = 3
# Number of measurements
M = 7

# Set the type of randomness for the entries of A
# type_A = 'N01'  # N(0, 1)
type_A = 'abs_N01'  # abs(N(0, 1))
#type_A = 'binary'  # (0, 1)
#type_A = 'uniform' # (A_min, A_max, dA)
params_A = (0.1, 1.01, 0.1) # used for unifrom A
# Set whether to normalize the columns of A to 1 in L2 norm
flag_norm_A = False

# Set the type of randomness for the solution vector x0
type_x0 = 'binary' # select from (0, 1)
#type_x0 = 'uniform' # select from (x0_min, x0_max, dx)
params_x0 = (0.1, 1.01, 0.25)

# Set whether and how much Gaussian noise N(0, 1) to add to the measurement
# vector
b_noise_coef = 0.2

# Set the coefficients of the sparsity regularizing term
lam_sparsity = np.exp(np.arange(+10, -10, -0.5))

# %% Generate the linear system

print("Generating the linear system")

[A, b, x0] = \
    RLS.generate_random_linear_system(N=N, M=M, K=K,
                                      type_A=type_A, params_A=params_A,
                                      flag_norm_A=flag_norm_A,
                                      type_x0=type_x0, params_x0=params_x0,
                                      b_noise_coef=b_noise_coef, seed=seed)

# %% Main loop

print("Going over lam_sparsity values")

s_vec = np.zeros(len(lam_sparsity))
err_system_vec = np.zeros(len(lam_sparsity))
err_system_lam_vec = np.zeros(len(lam_sparsity))
err_solution_fpn_vec = np.zeros(len(lam_sparsity))

for lam_idx, lam  in enumerate(lam_sparsity):

    print(" lam = ", lam)

    # Generate the QUBO matrix related to the current iteration
    # Q, h = annealer.generate_QUBO_linear_system_L0_sparsity_binary(A, b, lam)

    # Solve for the minimal energy using exhaustive search
    # spins, H = annealer.exhaustive_solve_QUBO_linear_system_L0_sparsity_binary(Q, h)
    
    # Solve the sparse linear system in QUBO formulation
    spins = annealer.solve_L0_binary_qubo(A, b, lam)
    
    # Set the number of spins
    num_spins = N

    # Form the Hamiltonian for the linear system of equations
    [Q1, h1] = annealer.generate_QUBO_linear_system(A, b.ravel())
    # Form the Hamiltonian for the sparsity (minimize cardinality)
    [Q2, h2] = annealer.generate_QUBO_L0_sparsity_binary(num_spins)

    # Combine the matrices into a single QUBO matrix, including weights
    Q = 1 * Q1 + lam * Q2
    H = 1 * h1 + lam * h2
    
    # Compute the solution error #(false positive) + #(false negative)
    err_solution_fpn = sum(abs(spins - x0))
    err_solution_fpn_vec[lam_idx] = err_solution_fpn
    # Compute the error according to the determinant of the confluence matrix
    #err_solution_cm = compute_confulence_matrix(x0, spins)

    s = int(sum(spins))
    s_vec[lam_idx] = s
    
    err_system = np.mean((A@spins - b)**2)
    err_system_vec[lam_idx] = err_system
    
    err_system_lam = H - lam * s
    err_system_lam_vec[lam_idx] = err_system_lam
    
    print("  ", spins, H, s, err_system, err_system_lam, err_solution_fpn)
    
# %% Plot results

plt.clf()
plt.title("Cardinality vs. lambda")
plt.plot(lam_sparsity, s_vec, 'x')
plt.xscale('log')
plt.xlabel("lambda")
plt.show()

plt.clf()
plt.title("System error vs. lambda")
plt.plot(lam_sparsity, err_system_vec, 'x')
plt.xscale('log')
plt.xlabel("lambda")
plt.show()

plt.clf()
plt.title("Solution error vs. lambda")
plt.plot(lam_sparsity, err_solution_fpn_vec, 'x')
plt.xscale('log')
plt.xlabel("lambda")
plt.show()

plt.clf()
plt.plot(s_vec, err_system_vec, 'x')
plt.title("System error vs. cardinality")
plt.xlabel("Number of non-zeros")
plt.show()

plt.clf()
plt.title("Solution error vs. cardinality")
plt.plot(s_vec, err_solution_fpn_vec, 'x')
plt.xlabel("Number of non-zeros")
plt.show()

