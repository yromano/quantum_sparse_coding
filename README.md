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

We provide Jupyter notebooks that cover a small-scale example, demonstrating the advantage of our QUBO formulation to the sparse coding problem. For demonstration purposes the code we provide runs on a classic computer. **It is important to emphasize that our QUBO problem can be minimized on quantum computers or other quantum-inspired solvers, see our paper [1] for larger-scale experiments. These are conducted on LightSolver's digital simulator [2].**

Binary sparse vectors: `Binary-Err-vs-NumSamples.ipynb` implements the proposed method with 1-bit representation for the unknown sparse $x$. We present the estimation error obtained as a function of the sample size.

2-bit sparse vectors: `Two-Bit-Err-vs-Noise.ipynb` implements the proposed method with 2-bit representation for the unknown sparse $x$. We present the estimation error obtained as a function of the noise level.

The CSV files available under `/results/` in the repository used to create the figures in the `/figures/` folder. See `/results/Plot_Graphs.py` for more details.

[2] I. Meirzada, A. Kalinski, D. Furman, T. Armon, T. Vaknin, H. Primack, C. Tradonsky, R. Ben Shlomi, "Lightsolver---a new quantum-
inspired solver cracks the 3-regular 3-XORSAT challenge," arXiv preprint arXiv:220709517, 2022.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
