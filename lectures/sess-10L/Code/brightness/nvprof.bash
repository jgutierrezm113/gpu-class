#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof.%j.out

cd /scratch/$USER/GPUClass18/HOL5/brightness/

set -o xtrace
nvprof ./brightness --image ../input/fractal.pgm
echo "METRICS"
nvprof -m all  ./brightness --image ../input/fractal.pgm

