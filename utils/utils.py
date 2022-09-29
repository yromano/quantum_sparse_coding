"""
A file with various general utility functions
Harel Primack
26.04.2022 v1.0: Created.
01.05.2022 v1.1: Added the function <compute_solution_error> to estimate the
                 quality of a solution obtained to the simulated solution.
"""

import numpy as np


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
#    x     = The obtained solution
#    x0    = The simulated solution
#    L0_th = Threshold for treating entries as zeros.

def compute_solution_error(x, x0, L0_th=1e-2):
    # Allocate and reset the output var
    s = np.ones(3) * np.NaN

    # Get the length of the solution vector for normalizations
    N = len(x)

    # Compute the difference between the solutions
    dx = x - x0

    # Compute the normalized L2 error
    s[2] = np.sqrt(sum(dx ** 2) / N)

    # Compute the normalized L1 error
    s[1] = sum(abs(dx)) / N

    # Compute the un-normalized L0 error =
    # #(false positive) + #(false negative)
    # We allow for a tolerance around zero
    # Compute the zero/non-zero entries of x
    L0_x = abs(x) > L0_th
    # Compute the zero/non-zero entries of x0
    L0_x0 = abs(x0) > L0_th
    # Sum the differences between the zero/non-zero entries
    s[0] = sum(L0_x != L0_x0)

    return s


# %%  For fixed point representation, apply validity checks,
#     and compute the spin combination that represents zero

def check_fixed_point_parameters(x_min, d_scale, P, verbose=False):

    if verbose:
        print(" Checking the validity of the fixed point parameters")

    # %% Check 1
    #    Check whether 0 is in the valid values of x_i
    # Span the values of x_i
    x_vals = x_min + d_scale * np.arange(2 ** P)

    # Check whether 0 is included in <x_vals>
    if 0 in x_vals:
        if verbose:
            print(" VVV Zero is a valid value of x_i")
            print(" x_vals =")
            print(" ", x_vals)
    else:
        print("XXX Error: Zero is not a valid value of x_i")
        print(" x_min =", x_min, "   d_scale =", d_scale)
        print(" x_vals =")
        print(" ", x_vals)
        1 / 0

    # Compute the value of the spin combination that represents 0,
    # and make sure it is an integer
    x_ip_0_real = -x_min / d_scale
    x_ip_0_int = int(x_ip_0_real)
    if abs(x_ip_0_real - x_ip_0_int) < 1e-12:
        if verbose:
            print(" VVV x_ip_0 is an integer =", x_ip_0_int)
    else:
        print(" Error: x_ip_0 is not an integer =", x_ip_0_real)
        1 / 0

    # Compute the binary representation of x_ip_0_int as a string
    x_ip_0_str = bin(x_ip_0_int)[2:].zfill(P)

    # Substitute in a binary array of spins
    x_ip_0_spins = np.zeros(P)
    for ind in range(P):
        x_ip_0_spins[ind] = int(x_ip_0_str[-ind - 1])

    if verbose:
        print(" x_ip_0 spins =", x_ip_0_spins)
        print(" Fixed point parameters are OK")

    return x_ip_0_spins