#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate base

python /home/yromano/sparse_coding/solvers_experiments/experiment_fp.py $A_id $N $M $K $Norm_A_id $X_id
