#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --tasks-per-node 8
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec.%j.out

cd /scratch/$USER/GPUClass18/HOL4/dp/

set -o xtrace
./baseline 100 3000
