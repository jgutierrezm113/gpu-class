#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof_shared
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof_shared.%j.out

cd /scratch/$USER/GPUClass18/HOL3/

set -o xtrace
nvprof --metrics all ./stencil_shared 1000000 

