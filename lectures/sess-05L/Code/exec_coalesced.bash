#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_coalesced
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec_coalesced.%j.out

cd /scratch/$USER/GPUClass18/HOL2/

set -o xtrace
./vadd_coalesced 100000000
