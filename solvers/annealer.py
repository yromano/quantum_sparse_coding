#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

base_code_path = './'

sys.path.append(base_code_path)
sys.path.append(base_code_path + "/solvers")
sys.path.append(base_code_path + "/utils")

import numpy as np
from tqdm import tqdm
from solvers import qubo_matrices as QM
from solvers import qubo_solvers as QS
from utils import helper


"""
QUBO solver for ||Ax-b||_2^2 + w_sparsity*||x||_norm

A    : the design matrix
b    : response vector
norm : "L0" or "L1"
params["x0_min"]      : minimal value of the solution
params["d_x0"]       : step for computing the valid values of the solution
params["num_bits"]   : fixed point resolution
params["w_fp"]       : the weight for the hard conditions (F-functions) in the sparsity computation
w_sparsity : the weight of the sparsity constraint in the QUBO matrix

"""
def solve_Lp_qubo(A, b, norm, params, w_sparsity):

    M, N = A.shape

    # Compute the combined QUBO matrix for the linear system and for the
    # sparsity condition
    if norm == "L0":
        Q, h = QM.generate_QUBO_linear_system_L0_sparsity_fixed_point(
                A, b, params["x0_min"], params["d_x0"], params["num_bits"], w_sparsity, params["w_fp"])
    else:
        print("norm can be only L0")
        0/1

    # Apply exhaustive search, and find the minimal solution per the current
    # lambda
    spins, H = QS.solve_QUBO_exhaustive(Q, h, display=False)

    return helper.convert_spins_to_solution(spins, N, params["x0_min"], params["num_bits"], params["d_x0"])


class qubo_Lp:
    def __init__(self, norm, params, reg_param):
        self.norm = norm
        self.params = params
        self.reg_param=reg_param
        self.coef_ = 0

    def fit(self,X,y):
        beta = solve_Lp_qubo(X,
                             y,
                             self.norm,
                             self.params,
                             w_sparsity=self.reg_param)
        self.coef_ = beta

    def predict(self,X):
        return np.dot(X,self.coef_[:,np.newaxis])


"""
Solve ||Ax-b||_2^2 + lambda*||x||_norm using QUBO formulation
A : the design matrix
b : response vector
A_test : test design matrix
b_test : test response vector
norm   : L0 or L1
params : fixed point solver internal hyper-params
lam    : list of possible regularization parameters 'lambda'
L0_th  : threshold over the estimated coeffs
x0     : true solution
"""
def exhaustive_qubo_Lp(A, b, A_test=None, b_test=None,
                       norm="L0", params=dict(), lam_Lp=0.1, L0_th = 1e-2, x0=None):

    # Extract dimentions
    M, N = A.shape

    # Prepare the output arrays for the solution
    num_cases_Lp = len(lam_Lp)
    x_Lp = np.zeros((num_cases_Lp, N))  # solution
    r_Lp = np.zeros(num_cases_Lp)       # system L2 error per measurement
    r_Lp_test = np.zeros(num_cases_Lp)  # system test L2 error per measurement
    s_Lp = np.zeros((num_cases_Lp, 3))  # solution's L0, L1, L2 error
    k_Lp = np.zeros(num_cases_Lp)       # solution's cardinality

    # Iterate over the various lambda values, increasing sparsity
    for ind, lam in enumerate(lam_Lp):
        # Prepare the solver object
        model = qubo_Lp(norm, params, reg_param=lam)
        # Apply the solver on the current case
        model.fit(A, b)
        # Extract the current solution and save in an array
        x_Lp[ind, :] = model.coef_
        # Compute the L2 error per measurement for the current solution
        dd = A @ x_Lp[ind, :] - b
        # Save the error result into an array
        r_Lp[ind] = np.sqrt(sum(dd ** 2) / M)

        if A_test is not None:
            dd = A_test @ x_Lp[ind, :] - b_test
            r_Lp_test[ind] = np.sqrt(sum(dd ** 2) / M)

        # Compute the support size of the current solution
        # "Hard" definition - no threshold
        # k_Lp[ind] = sum(x_Lp[ind, :] != 0)
        # "Soft" definition - use a threshold
        k_Lp[ind] = sum(abs(x_Lp[ind, :]) > L0_th)

        # Compute the errors of the solution
        if x0 is not None:
            s_Lp[ind, :] = helper.compute_solution_error(x_Lp[ind, :], x0, L0_th)


    return x_Lp, r_Lp, r_Lp_test, k_Lp, s_Lp
