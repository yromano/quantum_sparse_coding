"""
Generate a random linear system of equations based on the input parameters.

Harel Primack, 25.07.2021
"""

import numpy as np
import random
from utils import utils

from find_A_low_coherence import gen_low_coherence

"""
Generate a linear system of equations with optional sparsity
Ax = b, where we randomize A and x, and compute b.
N               = number of variables / features
M               = number of measurements / samples
K               = number of non-zero variables (sparsity, cardinality)
type_A          = 'low_coherence'  : consturcts low coherence matrix
type_x0         = 'binary': x0[i] = 1 for i in the cardinality set of size K
                = 'fixed point': Each x0[i] for i in the cardinality set
                    of size K is selected from the set
                    x_i = x_min_0 + d_i * sum_p=1^P (2 ** (p-1))
                    excluding 0 with equal probability
params_x0       = for 'fixed point': (x_mix, d_scale, P) - notation as above.
b_noise_coef    = Add to b noise b_noise_coef * N(0, 1) [default = 0]
seed            = Seed to the RNG [default = None]
"""


def generate_random_linear_system(
        N, M, K,
        type_A='low_coherence', type_x0='binary',
        params_x0=(),
        b_noise_coef=0,
        seed=None):
    # Set the seed of the RNG
    random.seed(a=seed)

    # Set the sampling matrix A
    # Reset the matrix
    A = np.zeros((M, N))
    # Make preparations for the uniform case
    if type_A == 'low_coherence':
        A = gen_low_coherence(M,N,seed)
    # Make preparations for the uniform case
    else:
        print('Invalid type_A = ', type_A)
        1 / 0
    # Normalize the columns of A to unity
    for ind_2 in range(N):
        col = A[:, ind_2]
        norm_L2 = np.sqrt(sum(col ** 2))
        A[:, ind_2] = A[:, ind_2] / norm_L2

    # Set the measurement vector (solution)
    # Reset the measurements vector
    x0 = np.zeros(N)
    # Make preparations for the fixed-point case
    if type_x0 == 'fixed_point':
        x_min, d_scale, P = params_x0
        x_ip_0_spins = utils.check_fixed_point_parameters(x_min, d_scale, P)
        x_vals_non_zero = []
        for ind in range(2 ** P):
            tmp = x_min + d_scale * ind
            if tmp != 0:
                x_vals_non_zero.append(tmp)

    # Select the random indices of the variables which are not zero
    inds_non_zero = random.sample(range(N), k=K)
    # Set the non-zero indices
    for ind in inds_non_zero:
        if type_x0 == 'binary':
            x0[ind] = 1
        elif type_x0 == 'fixed_point':
            x0[ind] = random.choice(x_vals_non_zero)
        else:
            print('Invalid type_x0 = ', type_x0)
            1 / 0

    # Generate the measurement vector
    b = A @ x0

    # Add noise if prescribed
    if b_noise_coef > 0:
        for ind in range(len(b)):
            b[ind] = b[ind] + b_noise_coef * random.gauss(mu=0, sigma=1)

    return A, b, x0
