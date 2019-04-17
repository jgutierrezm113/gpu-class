#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof_pinned
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof_pinned.%j.out

cd /scratch/$USER/GPUClassS19/HOL6/pinned/

set -o xtrace
nvprof ./vadd_pinned 100000000 

