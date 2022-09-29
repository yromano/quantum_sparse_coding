#!/bin/bash

#NORM_A_LIST=(0 1)
#A_LIST=(0 1 2 3)
#N_LIST=(16)
#M_LIST=(6 64)
#K_LIST=(1 2 3 4 5)
#X_LIST=(0) # X is binary

#NORM_A_LIST=(0)
#A_LIST=(1)
#N_LIST=(16)
#M_LIST=(6)
#K_LIST=(1 2 3 4 5)
#X_LIST=(0) # X is binary

#NORM_A_LIST=(1)
#A_LIST=(4)
#N_LIST=(16)
#M_LIST=(6 7 8 9 10 11 12 13 14 15 16)
#K_LIST=(1 2 3 4 5)
#X_LIST=(0) # X is binary


NORM_A_LIST=(1)
A_LIST=(4)
N_LIST=(160)
M_LIST=(60 70 80 90 100 110 120 130 140 150 160)
K_LIST=(10 20 30 40 50)
X_LIST=(0) # X is binary

# Slurm parameters
MEMO=16G              # Memory required (16GB)
TIME=14-00:00:00      # Time required (14 days)
CORE=1                # Cores required (1)

# Assemble order prefix
ORDP="--mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

LOGS="logs"
mkdir -p $LOGS

OUT_DIR="results_data/"
mkdir -p $OUT_DIR

for Norm_A_id in "${NORM_A_LIST[@]}"; do
  for A_id in "${A_LIST[@]}"; do
    for N in "${N_LIST[@]}"; do
      for M in "${M_LIST[@]}"; do
        for K in "${K_LIST[@]}"; do
          for X_id in "${X_LIST[@]}"; do
            # Define job name for this chromosome
            JOBN="A"$A_id"_N"$N"_M"$M"_K"$K"_NA"$Norm_A_id"_X"$X_id
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            sbatch -w socrates --mem=$MEMO --nodes=1 --ntasks=1 --cpus-per-task=1 --time=$TIME -J $JOBN -o $OUTF -e $ERRF --export=A_id=$A_id,N=$N,M=$M,K=$K,Norm_A_id=$Norm_A_id,X_id=$X_id /home/yromano/sparse_coding/solvers_experiments/submit.sh
          done
        done
      done
    done
  done
done
