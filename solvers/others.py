"""
Here we put classical solution algorithms for sparse regularization.
Classical = non-quantum.

Harel Primack, 27.02.2022
"""

import os
import sys

base_code_path = "./"

sys.path.append(base_code_path)
sys.path.append(base_code_path + "/solvers")
sys.path.append(base_code_path + "/utils")

import itertools
import numpy as np
from tqdm import tqdm
from utils import helper
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit, LassoCV, OrthogonalMatchingPursuitCV


"""
Classical solver for ||Ax-b||_2^2 s.t. ||x||_0 \leq k, for all k
The L0 regularization is done using exhaustive search

A : the design matrix
b : response vector
L0_th : threshold over the estimated coeffs
x0 : true solution
"""
def exhaustive_L0(A, b, A_test=None, b_test=None, L0_th=1e-2, x0=None, print_progress=False):

    # Extract useful quantities
    M, N = A.shape

    # Prepare the output arrays for the solution
    num_cases_L0 = N + 1
    x_L0 = np.zeros((num_cases_L0, N))  # solution
    r_L0 = np.ones(num_cases_L0) * np.Inf  # system's L2 error per measurement
    r_L0_test = np.ones(num_cases_L0) * np.Inf  # system's test L2 error per measurement
    s_L0 = np.zeros((num_cases_L0, 3))  # solution's error in L0, L1, L2 norms
    k_L0 = np.zeros(num_cases_L0)  # solution's cardinality

    # Construct all the L0 configurations
    L0_combinations = itertools.product([0, 1], repeat=N)

    # Go over all the L0 combinations
    for ind, comb in enumerate(L0_combinations):
        # Compute the cardinality of the current solution candidate
        k = sum(comb)

        # Print progress
        if print_progress and (ind % (2 ** (N - 4)) == 0):
            print(ind, comb, k)

        # Build the matrix that corresponds to the selected support -
        # select only the columns that correspond to the support
        inds_support = np.nonzero(comb)[0]
        AA = A[:, inds_support]

        # Solve the resulting system of equations in least square
        tmp = np.linalg.lstsq(AA, b, rcond=None)

        # Compute the L2 error per measurement of the linear system
        # for the current solution
        x_L0_tmp = tmp[0]
        dd = AA @ x_L0_tmp - b
        r_L0_tmp = np.sqrt(sum(dd ** 2) / M)

        # We keep for every cardinality the solution for which the system
        # error is minimal
        # Check whether the current solution has the minimal error per the
        # current cardinality
        if r_L0_tmp < r_L0[k]:
            # New minimum discovered, update the system error per the current
            # cardinality
            r_L0[k] = r_L0_tmp

            # Update the solution per the current support
            # Clean the vector
            x_L0[k, :] = 0
            # Update with the current solution
            x_L0[k, inds_support] = x_L0_tmp

            if A_test is not None:
                dd = A_test @ x_L0[k] - b_test
                r_L0_test[k] = np.sqrt(sum(dd ** 2) / M)

            # Compute the errors of the solution
            if x0 is not None:
                s_L0[k, :] = helper.compute_solution_error(x_L0[k, :], x0, L0_th)

            # Register the support size of the current solution
            # (this is trivial, but done to replicate the L1 data set)
            k_L0[k] = k

    return x_L0, r_L0, r_L0_test, k_L0, s_L0


"""
Solve ||Ax-b||_2^2 + lambda*||x||_1 using LASSO
A : the design matrix
b : response vector
lam_L1: list of possible regularization parameters 'lambda'
L0_th : threshold over the estimated coeffs
x0 : true solution
"""
def lasso_L1(A, b, A_test=None, b_test=None, flag_norm_A=False, L0_th = 1e-2, x0=None):

    if flag_norm_A == True:
        A_norm, norm_vec = helper.normalize_cols(A.copy())
    else:
        A_norm = A
        norm_vec = 1

    lambda_max = sum(np.abs(A_norm.T@b))
    eps = 1e-8
    n_grid = 1000
    lam_L1 = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_grid)[::-1]
    np.append(lam_L1, 1e-10)
    np.append(lam_L1, 1e-15)

    # Extract dimentions
    M, N = A.shape

    # Prepare the output arrays for the solution
    num_cases_L1 = len(lam_L1)
    x_L1 = np.zeros((num_cases_L1, N))  # solution
    r_L1 = np.zeros(num_cases_L1)       # system L2 error per measurement
    r_L1_test = np.zeros(num_cases_L1)  # system test L2 error per measurement

    s_L1 = np.zeros((num_cases_L1, 3))  # solution's L0, L1, L2 error
    k_L1 = np.zeros(num_cases_L1)       # solution's cardinality

    # Iterate over the various lambda values, increasing sparsity
    for ind, lam in enumerate(lam_L1):
        # Prepare the LASSO object
        lam_eff = lam / (2 * M)
        reg = Lasso(alpha=lam_eff, fit_intercept=False)
        # Apply the LASSO algorithm on the current case
        reg.fit(A_norm, b)
        # Extract the current solution and save in an array
        tmp_coeff = reg.coef_
        # Multiply by norm_vec
        tmp_coeff = tmp_coeff/norm_vec

        x_L1[ind, :] = tmp_coeff
        # Compute the L2 error per measurement for the current solution
        dd = A @ x_L1[ind, :] - b
        # Save the error result into an array
        r_L1[ind] = np.sqrt(sum(dd ** 2) / M)

        if A_test is not None:
            dd = A_test @ x_L1[ind, :] - b_test
            r_L1_test[ind] = np.sqrt(sum(dd ** 2) / M)


        # Compute the support size of the current solution
        # "Hard" definition - no threshold
        # k_L1[ind] = sum(x_L1[ind, :] != 0)
        # "Soft" definition - use a threshold
        k_L1[ind] = sum(abs(x_L1[ind, :]) > L0_th)

        # Compute the distance (error) between the current solution and the
        # original one
        if x0 is not None:
            s_L1[ind, :] = helper.compute_solution_error(x_L1[ind, :], x0, L0_th)

    return x_L1, r_L1, r_L1_test, k_L1, s_L1

"""
Solve ||Ax-b||_2^2 s.t. ||x||_0 \leq k, for all k, using OMP
A : the design matrix
b : response vector
L0_th : threshold over the estimated coeffs
x0 : true solution
"""
def omp_L0(A, b, A_test=None, b_test=None, L0_th=1e-2, x0=None):

    A_norm, norm_vec = helper.normalize_cols(A.copy())

    # Extract dimentions
    M, N = A.shape

    # Maximal number of non-zeros
    max_k = min(M,N)

    # Prepare the output arrays for the solution
    num_cases_L0 = max_k + 1

    x_omp = np.zeros((num_cases_L0, N))    # solution
    r_omp = np.ones(num_cases_L0) * 1e100  # system's L2 error per measurement
    r_omp_test  = np.ones(num_cases_L0) * 1e100  # system's test L2 error per measurement

    s_omp = np.zeros((num_cases_L0, 3))    # solution's L0, L1, L2 error
    k_omp = np.zeros(num_cases_L0)         # solution's cardinality

    # Iterate over the various lambda values, increasing sparsity
    for ind in range(num_cases_L0):
        if ind>0:
            # Prepare the OMP object
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=min(ind,max_k),
                                            normalize=False, fit_intercept=False)
            # Apply the LASSO algorithm on the current case
            reg.fit(A_norm, b)
            # Extract the current solution and save in an array
            tmp_coeff = reg.coef_
            # Multiply by norm_vec
            tmp_coeff = tmp_coeff/norm_vec

            # Extract the current solution and save in an array
            x_omp[ind, :] = tmp_coeff

        # Compute the L2 error per measurement for the current solution
        dd = A @ x_omp[ind, :] - b
        # Save the error result into an array
        r_omp[ind] = np.sqrt(sum(dd ** 2) / M)

        if A_test is not None:
            dd = A_test @ x_omp[ind, :] - b_test
            r_omp_test[ind] = np.sqrt(sum(dd ** 2) / M)

        # Compute the support size of the current solution
        # "Hard" definition - no threshold
        # k_L1[ind] = sum(x_L1[ind, :] != 0)
        # "Soft" definition - use a threshold
        k_omp[ind] = sum(abs(x_omp[ind, :]) > L0_th)

        # Compute the distance (error) between the current solution and the
        # original one
        if x0 is not None:
            s_omp[ind, :] = helper.compute_solution_error(x_omp[ind, :], x0, L0_th)


    return x_omp, r_omp, r_omp_test, k_omp, s_omp
