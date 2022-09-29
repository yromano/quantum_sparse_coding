
import numpy as np


def convert_spins_to_solution(spins, N, x0_min, num_bits, d_x0):
    # Compose the actual solution from the spin solution
    x_QUBO = np.zeros(N)
    for ind_spin in range(N):
        # Set the minimal value
        x_QUBO[ind_spin] = x0_min
        # Add the bit representation
        for ind_bit in range(num_bits):
            ind = ind_bit + num_bits * ind_spin
            x_QUBO[ind_spin] += d_x0 * (2 ** ind_bit) * spins[ind]
    return x_QUBO




# %% Convert a vector of real numbers into a vector of bits in fixed-point
#    representation

def convert_real_into_fixed_point(x, num_bits_per_var, x0_min, d_x0):
    # Get the length of the vector
    N = len(x)
    # Allocate and reset the output vector
    x0_fp_bits = np.zeros(N * num_bits_per_var) * np.NaN

    # Loop over the entries in the input vector and decompose to FP bits
    for ind_var in range(N):
        # Translate the real value of x_i into a binary number
        # Compute the integer number to convert to binary
        bb = round((x[ind_var] - x0_min) / d_x0)
        # Convert to binary, and make sure that the binary number will have the
        # correct number of bits
        tmp = bin(bb)[2:].zfill(num_bits_per_var)
        # Fill in the entries in the output vector related to the current
        # variable
        for ind_bit in range(num_bits_per_var):
            ind_spin = ind_bit + num_bits_per_var * ind_var
            x0_fp_bits[ind_spin] = int(tmp[-ind_bit - 1])

    return x0_fp_bits


# %% Convert a vector of bits in fixed-point representation into
# a vector of real numbers

def convert_fixed_point_into_real(v_bits, num_bits_per_var, x0_min, d_x0):
    # Compute the number of real-valued variables
    tmp = len(v_bits)
    N = int(tmp / num_bits_per_var)

    # Allocate and reset the output vector
    x = np.zeros(N)

    for ind_var in range(N):
        # Set the minimal value
        x[ind_var] = x0_min
        # Add the bit representation
        for ind_bit in range(num_bits_per_var):
            # Find the index of the current bit to consider
            ind = ind_bit + num_bits_per_var * ind_var
            x[ind_var] += d_x0 * (2 ** ind_bit) * v_bits[ind]

    return x


# %% Compute the error of a solution with respect to the simulated solution
#    using various metrics
def compute_solution_error(x, x0, L0_th=1e-2):
    # Allocate and reset the output var
    s = np.ones(3) * np.NaN

    # Get the length of the solution vector for normalizations
    N = len(x)

    # Compute the difference between the solutions
    dx = x - x0

    # Compute the normalized L2 error
    #s[2] = np.sqrt(sum(dx ** 2) / N)
    s[2] = np.sqrt(sum(dx ** 2) / sum(x0 ** 2))

    # Compute the normalized L1 error
    #s[1] = sum(abs(dx)) / N
    s[1] = sum(abs(dx)) / sum(abs(x0))

    # Compute the un-normalized L0 error =
    # #(false positive) + #(false negative)
    # We allow for a tolerance around zero
    # Compute the zero/non-zero entries of x
    L0_x = abs(x) > L0_th
    # Compute the zero/non-zero entries of x0
    L0_x0 = abs(x0) > L0_th
    # Sum the differences between the zero/non-zero entries
    #s[0] = sum(L0_x != L0_x0)
    s[0] = sum(L0_x != L0_x0)

    return s

def normalize_cols(A):
    M, N = A.shape
    norm_L2_vec = np.zeros(N)
    for ind_2 in range(N):
        col = A[:, ind_2]
        norm_L2 = np.sqrt(sum(col ** 2))
        A[:, ind_2] = A[:, ind_2] / norm_L2
        norm_L2_vec[ind_2] = norm_L2
    return A, norm_L2_vec