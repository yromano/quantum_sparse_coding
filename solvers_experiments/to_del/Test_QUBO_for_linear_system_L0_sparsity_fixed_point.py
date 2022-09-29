"""
Apply a comprehensive test of the QUBO matrix that is generated to describe
a linear system Ax = b with L0 sparsity regularization for variables x_i
which are of fixed point type:
min_x(|Ax - b|^2 + lambda * sum_i=1^N(x_i != 0)),
where
x_i = d * sum_p=1^P x_ip * (2 ** (p-1))

We simulate a linear system with prescribed features, and compare to the
solution obtained by the QUBO matrix.

We test the combined QUBU matrix:
Q = Q_linear_system + lambda * Q_cardinality
and look for a spin configuration that minimizes Q for a given lambda.

Note: The general form of fixed point vector is:
x_i = x_min_i + d_i * sum_p=1^P x_ip * (2 ** (p-1))
For brevity we set x_min_i = 0 for all i, and d_i = d for all i.

Harel Primack, 22.03.2022
"""

import random
import numpy as np
import random_linear_system as RLS
import regularization_classical as RC
import qubo_matrices as QM
import qubo_solvers as QS
import itertools
import utils

# %% Set parameters

# Set a seed for RNG applications

seed = None
# seed = 0

# Set the number of variables
N = 4
# Set the number of non-zero variables (cardinality)
K = 3
# Set Number of measurements
M = 5

# Set the type of randomness for the entries of the matrix A
# type_A = 'N01'  # N(0, 1)
type_A = 'abs_N01'  # abs(N(0, 1))
# type_A = 'binary'  # (0, 1)
params_A = []
# type_A = 'uniform' # (A_min, A_max, dA)
# params_A = (0.1, 1.01, 0.1)  # used for uniform A
# Set whether to normalize the columns of A to 1 in L2 norm
flag_norm_A = False

# Set the type of randomness for the solution vector x0
# type_x0 = 'binary' # select from (0, 1)
# Select x0 values from [x0_min : dx : x0_max]
type_x0 = 'uniform discrete'
# Set the parameters for uniform discrete
# The values of the solution must include 0 and their number must be
# a positive power of 2
x0_min   = 0
x0_max   = 1
num_bits = 4

# Set whether and how much Gaussian noise N(0, 1) to add to the measurement
# vector
b_noise_coef = 0.0

# Set the weight for the hard conditions (F-functions) in the sparsity
# computation
w_F = 10

# Set the values of the lambda parameter
lambda_vals = np.exp(np.arange(3, -10, -0.5))

# %% Init the random engine

random.seed(a=seed)

# %% Process the uniform discrete parameters into a form to be used below

# Compute the step for computing the valid values of x0
d_x0 = (x0_max - x0_min) / (2 ** num_bits - 1)

# Check that the values of x0 are valid
# Spell out explicitly the values of x0
x0_vals = np.arange(x0_min, x0_max + 1e-10, d_x0)

# Make sure that the value 0 is included in the values of x0
if any(x0_vals == 0):
    print("The uniform discrete values of x0 are valid =")
    print(x0_vals)
else:
    print("Error: The value 0 is not included in the values of x0")
    1 / 0

params_x0 = (x0_min, x0_max + 1e-10, d_x0)

# %% Display info

print("N =", N, ",  K =", K, ",  M = ", M)
print("x0_min =", x0_min, ",  x0_max =", x0_max, ',  dx0 =', d_x0)

# %% Generate a randomized linear system

print("Generating the linear system")

[A_LS, b_LS, x0_LS] = \
    RLS.generate_random_linear_system(N=N, M=M, K=K,
                                      type_A=type_A, params_A=params_A,
                                      flag_norm_A=flag_norm_A,
                                      type_x0=type_x0, params_x0=params_x0,
                                      b_noise_coef=b_noise_coef, seed=seed)

# %% Convert the simulated solution into fixed-point representation of spins
#    as a preparation for steps below

print("Converting the simulated solution into fixed-point spins")

x0_spins_bits = utils.convert_real_into_fixed_point(
                    x0_LS, num_bits, x0_min, d_x0)

# %% Direct verification 1:
#    Verify that the spin bits operating on the QUBO matrix that is related
#    to the linear system (without the sparsity condition) give zero as
#    required

print("\nDirect verification 1: Linear system")

print("Generating the QUBO matrix related to the linear system")

Q_LS, h_LS = QM.generate_QUBO_linear_system_fixed_point(
                A_LS, b_LS, x0_min, d_x0, num_bits)

print("Verifying the spin fixed-point solution for the QUBO matrix of the "
      "linear system")

s = x0_spins_bits @ Q_LS @ x0_spins_bits + h_LS

ok = (s < 1e-10)
if ok:
    print(" vvv x0_spins_bits verified linear system OK, s =", s)
else:
    print(" xxx x0_spins_bits is wrong for linear system, s =", s)
    1/0

# %% Direct verification 2:
#    Verify that the spin bits operating on the QUBO matrix that is related
#    to the L0 norm (without the linear system) give the correct cardinality

print("\nDirect verification 2: L0 norm")

print("Generating the QUBO matrix related to the L0 norm (cardinality)")

Q_L0, h_L0 = QM.generate_QUBO_L0_sparsity_fixed_point(x0_min, N, num_bits, w_F)

print("Verifying the spin fixed-point solution for the QUBO matrix of the "
      "L0 norm (cardinality)")

# Since Q_L0 includes ancilla spins, we need to go over all the combinations
# of the ancilla spins, and take the one that minimizes xQ_L0x + h_L0
if num_bits <= 2:
    num_ancilla_spins = 0
else:
    num_ancilla_spins = N * (num_bits - 2)
print(" The number of ancilla spins =", num_ancilla_spins)
ancilla_spins_combinations = itertools.product([0, 1], repeat=num_ancilla_spins)

r = np.Inf
for ind, spins_ancilla_candidate in enumerate(ancilla_spins_combinations):
    spins_candidate = \
        np.concatenate((x0_spins_bits, np.array(spins_ancilla_candidate)))
    tmp = (spins_candidate @ Q_L0 @ spins_candidate) + h_L0
    if tmp < r:
        r = tmp
        spins_solution = spins_candidate

d = abs(r - K)
ok = (d < 1e-10)
if ok:
    print(" vvv x0_spins_bits verified L0 cardinality OK,"
          "  K =", K, "  r =", r, "  d =", d)
else:
    print(" xxx x0_spins_bits is wrong for L0 cardinality,"
          "  K =", K, "  r =", r, "  d =", d)
    1/0


# %% Inverse verification
#    Loop over lambda values and compute the minimum for the resulting QUBO
#    matrix, compare to the simulated solution

print("\nInvestigating the inverse problem - "
      "going over lambda values with QUBO")

# Allocate and reset variables to save for later analysis
# We register the solutions and related values when the cardinality changes
# as a function of lambda
if num_bits <= 2:
    num_spins = N * num_bits
else:
    num_spins = N * (2 * num_bits - 2)
print(" The total number of spins = ", num_spins)

QUBO_lambda_critical = np.zeros(K+1) * np.NaN
QUBO_spin_solutions  = np.zeros(((K+1), num_spins)) * np.NaN
QUBO_solutions       = np.zeros(((K+1), N)) * np.NaN
QUBO_solution_error  = np.zeros((K+1)) * np.NaN
QUBO_system_error    = np.zeros((K+1)) * np.NaN
# Keep a variable to remember the cardinality of the previous iteration to
# identify cardinality changes
cardinality_old = (-1)

for lam in lambda_vals:

    print("\n lam =", lam)

    # Compute the combined QUBO matrix for the linear system and for the
    # sparsity condition
    Q, h = QM.generate_QUBO_linear_system_L0_sparsity_fixed_point(
            A_LS, b_LS, x0_min, d_x0, num_bits, lam, w_F)

    # Apply exhaustive search, and find the minimal solution per the current
    # lambda
    print(" Applying exhaustive search over QUBO")
    spins, H_QUBO = QS.solve_QUBO_exhaustive(Q, h, display=False)

    # Compose the actual, real-valued solution from the spin solution
    x_QUBO = utils.convert_fixed_point_into_real(
        spins[0:(N * num_bits)], num_bits, x0_min, d_x0)

    # Compute the cardinality of x_QUBO
    cardinality = int(sum(x_QUBO != 0))

    # Compute the system error
    err_system = H_QUBO - lam * cardinality

    # Compute the solution error in L1 (FP + FN)
    z1 = (x_QUBO == 0)
    z2 = (x0_LS  == 0)
    err_solution_fpn = sum(z1 != z2)

    # Save values if the cardinality has changed from the previous iteration
    if cardinality_old != cardinality:
        print(" * Critical values updating")
        QUBO_lambda_critical[cardinality]   = lam
        QUBO_spin_solutions[cardinality, :] = spins
        QUBO_solutions[cardinality, :]      = x_QUBO
        QUBO_solution_error[cardinality]    = err_solution_fpn
        QUBO_system_error[cardinality]      = err_system
        # Update the new cardinality
        cardinality_old = cardinality

    # Compare the current solution to the simulated solution
    print(" x0_LS  =", x0_LS)
    print(" x_QUBO =", x_QUBO)
    dd = x0_LS - x_QUBO
    print(" dd     =", dd)

    if np.all(dd == 0):
        print(" Inverse verified OK")
        break
    else:
        print(" Decreasing lambda")

if not np.all(dd == 0):
    print(" Inverse verified FAILED")

# %% Solve classically to verify that there are no redundant solutions and to
#    compare to the lambda-solutions of the QUBO system

if 1:
    print("\nSolving the linear system classically using L0 regularization")
    x_L0, r_L0, s_L0, k_L0 = RC.Regularize_L0(A_LS, b_LS, x0_LS, L0_th=1e-2)

    print("\nComparing classical vs. QUBO solutions per cardinality")
    for cardinality in range(K+1):
        print("\n cardinality =", cardinality)
        print(" r_L0   =", r_L0[cardinality])
        print(" s_L0   =", s_L0[cardinality])
        print(" k_L0   =", k_L0[cardinality])
        print(" x_L0   =", x_L0[cardinality])
        print(" x_QUBO =", QUBO_solutions[cardinality])
        print(" x0_LS  =", x0_LS)


