#!/bin/bash

#NORM_A_LIST=(0 1)
#A_LIST=(0 1 2 3)
#N_LIST=(6)
#M_LIST=(4 24)
#K_LIST=(1 2 3)
#X_LIST=(1) # X is of 2 bits


NORM_A_LIST=(1)
A_LIST=(4)
N_LIST=(80)
M_LIST=(30 35 40 50 60 70)
K_LIST=(10)
X_LIST=(1)


# Slurm parameters
MEMO=16G              # Memory required (16GB)
TIME=14-00:00:00      # Time required (14 days)
CORE=1                # Corses required (1)
#W_NAME="socrates"
W_NAME="galileo1"

#W_NAME="newton3"
#W_NAME="lambda4"

# Assemble order prefix
ORDP="--mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

LOGS="logs_fp"
mkdir -p $LOGS

OUT_DIR="results_fp/"
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
            sbatch -w $W_NAME --mem=$MEMO --nodes=1 --ntasks=1 --cpus-per-task=1 --time=$TIME -J $JOBN -o $OUTF -e $ERRF --export=A_id=$A_id,N=$N,M=$M,K=$K,Norm_A_id=$Norm_A_id,X_id=$X_id /home/yromano/sparse_coding/solvers_experiments/submit.sh
          done
        done
      done
    done
  done
done
