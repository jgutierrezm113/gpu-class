#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_shared
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec_shared.%j.out

cd /scratch/$USER/GPUClass18/HOL3/

set -o xtrace
./stencil_shared 1000
./stencil_shared 100000
./stencil_shared 1000000
./stencil_shared 10000000
./stencil_shared 100000000
