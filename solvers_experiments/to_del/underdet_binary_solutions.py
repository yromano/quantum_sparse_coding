"""
This is the main script to run the sparse regularization algorithms for
*binary* solutions of linear systems.

Harel Primack, 27.02.2022
"""

import numpy as np
import pandas as pd

import sys

base_code_path = "/Users/romano/Library/CloudStorage/OneDrive-Technion/drive/lightsolver/repo_sparse/sparse_coding"
sys.path.append(base_code_path)
sys.path.append(base_code_path + "/solvers")
sys.path.append(base_code_path + "/utils")

from utils import Plot_Graphs as PG
import random_linear_system as RLS
from solvers import annealer
from solvers import others

def get_stats(s_L, r_L, r_L_test):
    # Find the best solution, minimizing the L2 distance between x_hat and x_0
    tmp = s_L[:,2]
    ind_best_solution_x = np.argmin(tmp)
    rec_err_x = tmp[ind_best_solution_x]
    cardinality_err_x = s_L[ind_best_solution_x,0]
    rmse = r_L[ind_best_solution_x]
    
    # Find the best CV solution, minimizing the test system error
    ind_best_solution_cv_sys = np.argmin(r_L_test)
    cv_rec_err_x = s_L[ind_best_solution_cv_sys,2]
    cv_cardinality_err_x = s_L[ind_best_solution_cv_sys,0]
    cv_rmse = r_L_test[ind_best_solution_cv_sys]
    
    return rec_err_x, cardinality_err_x, rmse, cv_rec_err_x, cv_cardinality_err_x, cv_rmse


# %% Parameters

# The seed for the RNG
seed = None
# seed = 0

# Number of (features) variables
N = 16
# Number of non-zero variables (cardinality)
K_vec = [1, 2, 3, 4, 5]
K_vec = [3]
# Number of measurements
M = 6

# Set the type of randomness for the entries of A
type_A = 'N01'  # N(0, 1)
# type_A = 'abs_N01'  # abs(N(0, 1))
# params_A = ()
# type_A = 'uniform' # (A_min, A_max, dA)
params_A = (0.1, 1.01, 0.1) # used for unifrom A
# Set whether to normalize the columns of A to 1 in L2 norm
flag_norm_A = True

# Set the type of randomness for the solution vector x0
type_x0 = 'binary' # select from (0, 1)
params_x0 = ()
#type_x0 = 'uniform' # select from (x0_min, x0_max, dx)
# params_x0 = (0, 1.01, 0.25)

# Set whether and how much Gaussian noise N(0, 1) to add to the measurement
# vector
noise_vec = [0.0, 0.2, 0.4, 0.7, 1.0]
noise_vec = [0.0]

# Set the coefficients of the L1 regularizing term
lam_L1 = []
for p in np.arange(5, -5, -0.01):
    lam_L1.append(10 ** p)
lam_L1.append(1e-8)

# Set the threshold to the absolute value of and element of a solution
# vector, under which this element is considered as zero for cardinality
L0_th = 1e-2


# %% Run experiments

num_exps = 5

# Initialize table of results
results = pd.DataFrame()

for card_id, K  in enumerate(K_vec):
    for noise_id, b_noise_coef in enumerate(noise_vec):
        for exp_id in range(num_exps):
            print("card_id: " + str(card_id+1) + "/" + str(len(K_vec)) + " noise_id: " + str(noise_id+1) + "/" + str(len(noise_vec)) + " exp_id: " + str(exp_id+1) + "/" + str(num_exps))

            # %% Form the system of equations
            [A, b, x0, A_test, b_test] = \
                RLS.generate_random_linear_system(N=N, M=M, K=K,
                                                  type_A=type_A, params_A=params_A,
                                                  flag_norm_A=flag_norm_A,
                                                  type_x0=type_x0, params_x0=params_x0,
                                                  b_noise_coef=b_noise_coef, seed=seed)

            # %% Apply L0 regularization (classical)

            x_L0, r_L0, r_L0_test, k_L0, s_L0 = annealer.exhaustive_L0(A, b,
                                                                       A_test, b_test,
                                                                       L0_th=L0_th, x0=x0)
            
            rec_err_L0, cardinality_err_L0, rmse_L0, \
                cv_rec_err_L0, cv_cardinality_err_L0, cv_rmse_L0 = \
                    get_stats(s_L0, r_L0, r_L0_test)

            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'Exhaustive L0',
                                      'Best X-Recovery L2 Err':rec_err_L0,
                                      'Best X-Recovery L0 Err':cardinality_err_L0,
                                      'Best System RMSE':rmse_L0,
                                      'CV X-Recovery L2 Err':cv_rec_err_L0,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_L0,
                                      'CV System RMSE':cv_rmse_L0,
                                      'Type A':type_A,
                                      'Normalize A':flag_norm_A,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M},
                                      ignore_index=True)

            # %% Apply L0 regularization (QUBO)

            lambda_max = sum(np.abs(A.T@b))
            eps = 1e-8
            n_grid = 20
            lam_L0 = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_grid)[::-1]

            x_qubo, r_qubo, r_qubo_test, k_qubo, s_qubo = \
                annealer.exhaustive_qubo_binary_L0(A, b,
                                                   A_test, b_test,
                                                   lam_L0,
                                                   L0_th=L0_th, x0=x0)
            
            rec_err_qubo, cardinality_err_qubo, rmse_qubo, \
                cv_rec_err_qubo, cv_cardinality_err_qubo, cv_rmse_qubo = \
                    get_stats(s_qubo, r_qubo, r_qubo_test)

            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'QUBO L0',
                                      'Best X-Recovery L2 Err':rec_err_qubo,
                                      'Best X-Recovery L0 Err':cardinality_err_qubo,
                                      'Best System RMSE':rmse_qubo,
                                      'CV X-Recovery L2 Err':cv_rec_err_qubo,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_qubo,
                                      'CV System RMSE':cv_rmse_qubo,
                                      'Type A':type_A,
                                      'Normalize A':flag_norm_A,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M},
                                      ignore_index=True)



            # %%
            # Apply L1 regularization
            x_L1, r_L1, r_L1_test, k_L1, s_L1 = others.lasso_L1(A, b, 
                                                                A_test, b_test,
                                                                lam_L1, 
                                                                L0_th=L0_th, x0=x0)
            
            rec_err_L1, cardinality_err_L1, rmse_L1, \
                cv_rec_err_L1, cv_cardinality_err_L1, cv_rmse_L1 = \
                    get_stats(s_L1, r_L1, r_L1_test)

            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'Lasso',
                                      'Best X-Recovery L2 Err':rec_err_L1,
                                      'Best X-Recovery L0 Err':cardinality_err_L1,
                                      'Best System RMSE':rmse_L1,
                                      'CV X-Recovery L2 Err':cv_rec_err_L1,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_L1,
                                      'CV System RMSE':cv_rmse_L1,
                                      'Type A':type_A,
                                      'Normalize A':flag_norm_A,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M},
                                      ignore_index=True)

            # %%
            # Apply L0 regularization using OMP
            x_omp, r_omp, r_omp_test, k_omp, s_omp = others.omp_L0(A, b, 
                                                                   A_test, b_test,
                                                                   L0_th=L0_th, x0=x0)
            
            rec_err_omp, cardinality_err_omp, rmse_omp, \
                cv_rec_err_omp, cv_cardinality_err_omp, cv_rmse_omp = \
                    get_stats(s_omp, r_omp, r_omp_test)
                    
            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'OMP',
                                      'Best X-Recovery L2 Err':rec_err_omp,
                                      'Best X-Recovery L0 Err':cardinality_err_omp,
                                      'Best System RMSE':rmse_omp,
                                      'CV X-Recovery L2 Err':cv_rec_err_omp,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_omp,
                                      'CV System RMSE':cv_rmse_omp,
                                      'Type A':type_A,
                                      'Normalize A':flag_norm_A,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M},
                                      ignore_index=True)


# %% Save to CSV

results.to_csv("results_" + "type_x0" + type_x0 + "_type_A_" + type_A + "norm_A_" + str(flag_norm_A) + "_M_" + str(M) + "_N_" + str(N) + "_maxK_" + str(max(K_vec)) + "_maxNoise_" + str(max(noise_vec)) + ".csv")

# %% Plot results for classical regularizations

# Plot the accuracy (error) of the system and of the solution
PG.Plot_System_And_Solution_Accuracy(
        N, K, M, k_L0, r_L0, s_L0, k_L1, r_L1, s_L1, k_omp, r_omp, s_omp)


# %%
print(results)
