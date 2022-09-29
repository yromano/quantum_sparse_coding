# Quantum Sparse Coding


This package contains a Python implementation for Quantum Sparse Coding [1]: a quantum-inspired method for recovering a sparse vector given a few noisy linear measurements.

[1] Y. Romano, H. Primack, T. Vaknin, I. Meirzada, I. Karpas, D. Furman, C. Tradonsky, R. Ben Shlomi, "Quantum Sparse Coding," 2022. Link to paper: https://arxiv.org/abs/2209.03788

## Background

The ultimate goal of any sparse coding method is to accurately recover from a few noisy linear measurements, an unknown sparse vector. Unfortunately, this estimation problem is NP-hard in general, and it is therefore always approached with an approximation method, such as lasso or orthogonal matching pursuit, thus trading off accuracy for less computational complexity. In this package we implement a quantum-inspired algorithm for sparse coding, with the premise that the emergence of quantum computers and Ising machines can potentially lead to more accurate estimations compared to classical approximation methods. To this end, we formulate the most general sparse coding problem as a quadratic unconstrained binary optimization (QUBO) task, which can be efficiently minimized using quantum technology. To derive at a QUBO model that is also efficient in terms of the number of spins (space complexity), we separate our analysis into different cases, defined by the number of bits required to express the underlying sparse vector.

## Getting Started

This package is self-contained and implemented in python.

### Prerequisites

* python
* numpy
* scipy
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/yromano/quantum_sparse_coding.git
```

## Usage

### Jupyter notebooks

We provide Jupyter notebooks that include small-scale examples to demonstrate the advantage of our QUBO approach to the sparse coding problem. For demonstration purposes the code we provide runs on a classic computer. **It is important to emphasize that the proposed QUBO problem can be minimized on quantum computers or other quantum-inspired solvers. In our paper [1] we run this approach on large-scale problems using LightSolver's digital simulator [2].**

Binary sparse vectors: `Binary-Err-vs-NumSamples.ipynb` implements the proposed method with 1-bit representation for the unknown sparse $x$. We present the estimation error obtained as a function of the sample size.

2-bit sparse vectors: `Two-Bit-Err-vs-Noise.ipynb` implements the proposed method with 2-bit representation for the unknown sparse $x$. We present the estimation error obtained as a function of the noise level.

The CSV files available under `/results/` in the repository used to create the figures in the `/figures/` folder. See `/results/Plot_Graphs.py` for more details.

[2] I. Meirzada, A. Kalinski, D. Furman, T. Armon, T. Vaknin, H. Primack, C. Tradonsky, R. Ben Shlomi, "Lightsolver---a new quantum-
inspired solver cracks the 3-regular 3-XORSAT challenge," arXiv preprint arXiv:220709517, 2022.

### Generate data

To generate the data used in the motivating example (Figure 1 in [1]), run the function

```
generate_random_linear_system(N, M, K, type_A='low_coherence', type_x0, params_x0, b_noise_coef, seed)
```
from `/solvers_experiments/random_linear_system.py`. We set $N=16$; $M$ in the range 6,...,16; $K$ in the range 1,...,4; type_A="low_coherence"; type_x0="binary"; params_x0=(); 'b_noise_coef' in the range [0.0, 0.3]; and 'seed' in the range 1,...,20.



To generate the data used in the large scale 1-bit experiment (Figure 2 in [1]), run the function

```
generate_random_linear_system(N, M, K, type_A='low_coherence', type_x0, params_x0, b_noise_coef, seed)
```
from `/solvers_experiments/random_linear_system.py`. We set $N=160$; $M$ in the range 60,...,160; $K$ in the range 10,20,30; type_A="low_coherence"; type_x0="binary"; params_x0=(); 'b_noise_coef' in the range [0.0, 0.2]; and 'seed' in the range 1,...,20.


To generate the data used in the large scale 2-bit experiment (Figure 3 in [1]), run the function

```
generate_random_linear_system(N, M, K, type_A='low_coherence', type_x0, params_x0, b_noise_coef, seed)
```
from `/solvers_experiments/random_linear_system.py`. We set $N=80$; $M$ in the range 30,...,60; $K=10$; type_A="low_coherence"; type_x0="fixed_point"; params_x0=(0, 1, 2); 'b_noise_coef'$=0.1$; and 'seed' in the range 1,...,20.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
