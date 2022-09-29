import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys


if os.path.isdir('/home/yromano'):
    print("server")
    local_machine = 0
else:
    print("local machine")
    local_machine = 1
    
if local_machine:
    base_code_path = "/Users/romano/Library/CloudStorage/OneDrive-Technion/drive/lightsolver/repo_sparse/sparse_coding/"
else:
    base_code_path = '/home/yromano/sparse_coding'

sys.path.append(base_code_path)
sys.path.append(base_code_path + "/solvers")
sys.path.append(base_code_path + "/utils")

pd.set_option('display.max_columns', None)

#########################
# Experiment parameters #
#########################

#Type of A
type_A_id = 3
# Number of (features) variables
N = 100
# Number of measurements
M = 80
# Number of non-zero variables (cardinality)
K = 10

# internal normalization of A (lasso)
lasso_normalize = 1
# binary x0 or P-bits?
type_x0_id = 0

# Whether to normalize the columns of A to 1 in L2 norm
flag_norm_A = False

TYPE_A_LIST =  ["N01",
                "abs_N01",
                "binary",
                "uniform"]

TYPE_X_LIST =  ["binary",
                "uniform discrete"]



type_A = TYPE_A_LIST[type_A_id]
type_x0 = TYPE_X_LIST[type_x0_id]

print("Type A:\n  " + type_A)
print("Type x0:\n  " + type_x0)
sys.stdout.flush()


params_A = [0, 1, 0.25] # used for unifrom A

# Set the parameters for uniform discrete
# The values of the solution must include 0 and their number must be
# a positive power of 2
if type_x0 == "uniform discrete":
    x0_min = 0
    x0_max = 1
    num_bits = 2
    # Compute the step for computing the valid values of x0
    d_x0 = (x0_max - x0_min) / (2 ** num_bits - 1)
    w_fp = 50
    
    params_x0 = (x0_min, x0_max + 1e-10, d_x0)
elif type_x0 == "binary":
    x0_min = 0
    x0_max = 1
    num_bits = 1
    d_x0 = 1
    w_fp = 50
else:
    1/0

params_x0 = (x0_min, x0_max + 1e-10, d_x0)

print("N =", N, ",  K =", K, ",  M = ", M)
print("x0_min =", x0_min, ",  x0_max =", x0_max, ',  dx0 =', d_x0)


# arrrange params for the solver
params = dict()
params["x0_min"]   = x0_min
params["x0_max"]   = x0_max
params["num_bits"] = num_bits
params["d_x0"]     = d_x0
params["w_fp"]     = w_fp


# Set whether and how much Gaussian noise N(0, 1) to add to the measurement
# vector
# noise_vec = [0.0, 0.1, 0.2, 0.4, 0.7, 1.1, 1.5]
#noise_vec = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_vec = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

noise_vec = [0.1]

# Set the coefficients of the L1 regularizing term
# lam_L1 = []
# for p in np.arange(5, -5, -0.01):
#     lam_L1.append(10 ** p)
# lam_L1.append(1e-8)
# lam_L1.append(1e-10)


# Set the threshold to the absolute value of and element of a solution
# vector, under which this element is considered as zero for cardinality
#L0_th = 1e-2
L0_th = 1e-1

# number of trials per experiemnt
num_exps = 1

exp_name = "num_exps_" + str(num_exps) + "_type_x0_" + type_x0 + "_type_A_" + type_A + "_norm_A_" + str(flag_norm_A) + "_M_" + str(M) + "_N_" + str(N) + "_K_" + str(K) + "_lasso_normalize_" + str(lasso_normalize) + "_maxNoise_" + str(max(noise_vec))
print(exp_name)
sys.stdout.flush()


###################
# Output location #
###################

out_dir = base_code_path + "/solvers_experiments/results"
if not os.path.isdir(out_dir):
    print("creating results dir")
    os.makedirs(out_dir, exist_ok=True)
out_file = exp_name + ".csv"

print("Output directory for this experiment:\n  " + out_dir)
print("Output file for this experiment:\n  " + out_file)
out_file = out_dir + "/" + out_file


###################
# Core execution #
###################

import random_linear_system as RLS
from solvers import annealer
from solvers import others

def get_stats(s_L, r_L, r_L_test):
    # Find the best solution, minimizing the L2 distance between x_hat and x_0
    tmp = s_L[:,2]
    ind_best_solution_x = np.argmin(tmp)
    rec_err_x = tmp[ind_best_solution_x]
    rmse = r_L[ind_best_solution_x]
    
    # Find the best solution, minimizing the L0 distance between x_hat and x_0
    tmp = s_L[:,0]
    ind_best_solution_x = np.argmin(tmp)
    cardinality_err_x = s_L[ind_best_solution_x,0]
    
    
    # Find the best CV solution, minimizing the test system error
    ind_best_solution_cv_sys = np.argmin(r_L_test)
    cv_rec_err_x = s_L[ind_best_solution_cv_sys,2]
    cv_cardinality_err_x = s_L[ind_best_solution_cv_sys,0]
    cv_rmse = r_L_test[ind_best_solution_cv_sys]
    
    return rec_err_x, cardinality_err_x, rmse, cv_rec_err_x, cv_cardinality_err_x, cv_rmse


# Initialize table of results
results = pd.DataFrame()

run_exhaustive_L0 = False
run_exhaustive_qubo_L0 = False
run_lasso_L1 = True
run_exhaustive_qubo_L1 = False
run_omp_L0 = True

for noise_id, b_noise_coef in enumerate(noise_vec):
    for exp_id in range(num_exps):
        print("noise_id: " + str(noise_id+1) + "/" + str(len(noise_vec)) + " exp_id: " + str(exp_id+1) + "/" + str(num_exps))
        sys.stdout.flush()
        
        # %% Form the system of equations
        [A, b, x0, A_test, b_test] = \
            RLS.generate_random_linear_system(N=N, M=M, K=K,
                                              type_A=type_A, params_A=params_A,
                                              flag_norm_A=flag_norm_A,
                                              type_x0=type_x0, params_x0=params_x0,
                                              b_noise_coef=b_noise_coef, seed=exp_id)
        

        
        # %% Apply L0 regularization (classical)
        if run_exhaustive_L0 == True:
            x_L0, r_L0, r_L0_test, k_L0, s_L0 = others.exhaustive_L0(A, b,
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
                                      'Infeasible X-Recovery L2 Err':rec_err_L0,
                                      'Infeasible X-Recovery L0 Err':cardinality_err_L0,
                                      'Infeasible System RMSE':rmse_L0,
                                      'CV X-Recovery L2 Err':cv_rec_err_L0,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_L0,
                                      'CV System RMSE':cv_rmse_L0,
                                      'Type A':type_A,
                                      'Normalize A':lasso_normalize,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M,
                                      'x0_min':x0_min,
                                      'x0_max':x0_max,
                                      'num_bits':num_bits,
                                      'd_x0':d_x0,
                                      'w_fp':w_fp,
                                      'type_x0':type_x0},
                                      ignore_index=True)
        


        # %% Apply L0 regularization (QUBO)
        if run_exhaustive_qubo_L0 == True:
            lambda_max = sum(np.abs(A.T@b))
            eps = 1e-8
            n_grid = 20
            lam_L0 = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_grid)[::-1]
            np.append(lam_L0, 1e-10)
            np.append(lam_L0, 1e-15)
            
            
            x_qubo, r_qubo, r_qubo_test, k_qubo, s_qubo = \
                annealer.exhaustive_qubo_Lp(A, b,
                                            A_test, b_test,
                                            norm="L0",
                                            params=params,
                                            lam_Lp=lam_L0,
                                            L0_th=L0_th, x0=x0)
            
                
                
            rec_err_qubo, cardinality_err_qubo, rmse_qubo, \
                cv_rec_err_qubo, cv_cardinality_err_qubo, cv_rmse_qubo = \
                    get_stats(s_qubo, r_qubo, r_qubo_test)
    
            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'QUBO L0',
                                      'Infeasible X-Recovery L2 Err':rec_err_qubo,
                                      'Infeasible X-Recovery L0 Err':cardinality_err_qubo,
                                      'Infeasible System RMSE':rmse_qubo,
                                      'CV X-Recovery L2 Err':cv_rec_err_qubo,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_qubo,
                                      'CV System RMSE':cv_rmse_qubo,
                                      'Type A':type_A,
                                      'Normalize A':lasso_normalize,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M,
                                      'x0_min':x0_min,
                                      'x0_max':x0_max,
                                      'num_bits':num_bits,
                                      'd_x0':d_x0,
                                      'w_fp':w_fp,
                                      'type_x0':type_x0},
                                      ignore_index=True)



        # %%
        # lambda_max = sum(np.abs(A.T@b))
        # eps = 1e-8
        # n_grid = 1000
        # lam_L1 = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_grid)[::-1]
        # np.append(lam_L1, 1e-10)
        if run_lasso_L1 == True:
            # Apply L1 regularization
            x_L1, r_L1, r_L1_test, k_L1, s_L1 = others.lasso_L1(A, b, 
                                                                A_test, b_test,
                                                                lasso_normalize, 
                                                                L0_th=L0_th, x0=x0)
            
            rec_err_L1, cardinality_err_L1, rmse_L1, \
                cv_rec_err_L1, cv_cardinality_err_L1, cv_rmse_L1 = \
                    get_stats(s_L1, r_L1, r_L1_test)
    
            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'Lasso',
                                      'Infeasible X-Recovery L2 Err':rec_err_L1,
                                      'Infeasible X-Recovery L0 Err':cardinality_err_L1,
                                      'Infeasible System RMSE':rmse_L1,
                                      'CV X-Recovery L2 Err':cv_rec_err_L1,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_L1,
                                      'CV System RMSE':cv_rmse_L1,
                                      'Type A':type_A,
                                      'Normalize A':lasso_normalize,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M,
                                      'x0_min':x0_min,
                                      'x0_max':x0_max,
                                      'num_bits':num_bits,
                                      'd_x0':d_x0,
                                      'w_fp':w_fp,
                                      'type_x0':type_x0},
                                      ignore_index=True)

       
        # %% Apply L1 regularization (QUBO)
        if run_exhaustive_qubo_L1:
            lambda_max = sum(np.abs(A.T@b))
            eps = 1e-8
            n_grid = 20
            lam_L1 = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_grid)[::-1]
            np.append(lam_L1, 1e-10)
            np.append(lam_L1, 1e-15)
    
            x_qubo, r_qubo, r_qubo_test, k_qubo, s_qubo = \
                annealer.exhaustive_qubo_Lp(A, b,
                                            A_test, b_test,
                                            norm="L1",
                                            params=params,
                                            lam_Lp=lam_L1,
                                            L0_th=L0_th, x0=x0)
            
                
                
            rec_err_qubo, cardinality_err_qubo, rmse_qubo, \
                cv_rec_err_qubo, cv_cardinality_err_qubo, cv_rmse_qubo = \
                    get_stats(s_qubo, r_qubo, r_qubo_test)
    
            # Store results
            results = results.append({'Experiment':exp_id,
                                      'Cardinality': K,
                                      'Noise STD': b_noise_coef,
                                      'Method':'QUBO L1',
                                      'Infeasible X-Recovery L2 Err':rec_err_qubo,
                                      'Infeasible X-Recovery L0 Err':cardinality_err_qubo,
                                      'Infeasible System RMSE':rmse_qubo,
                                      'CV X-Recovery L2 Err':cv_rec_err_qubo,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_qubo,
                                      'CV System RMSE':cv_rmse_qubo,
                                      'Type A':type_A,
                                      'Normalize A':lasso_normalize,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M,
                                      'x0_min':x0_min,
                                      'x0_max':x0_max,
                                      'num_bits':num_bits,
                                      'd_x0':d_x0,
                                      'w_fp':w_fp,
                                      'type_x0':type_x0},
                                      ignore_index=True)
        
        # %%
        # Apply L0 regularization using OMP
        if run_omp_L0 == True:
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
                                      'Infeasible X-Recovery L2 Err':rec_err_omp,
                                      'Infeasible X-Recovery L0 Err':cardinality_err_omp,
                                      'Infeasible System RMSE':rmse_omp,
                                      'CV X-Recovery L2 Err':cv_rec_err_omp,
                                      'CV X-Recovery L0 Err':cv_cardinality_err_omp,
                                      'CV System RMSE':cv_rmse_omp,
                                      'Type A':type_A,
                                      'Normalize A':lasso_normalize,
                                      'L0 Thresh':L0_th,
                                      'N':N,
                                      'M':M,
                                      'x0_min':x0_min,
                                      'x0_max':x0_max,
                                      'num_bits':num_bits,
                                      'd_x0':d_x0,
                                      'w_fp':w_fp,
                                      'type_x0':type_x0},
                                      ignore_index=True)


# Save results on file
if out_file is not None:
    results.to_csv(out_file)


print("Output file for this experiment:\n  " + out_file)
