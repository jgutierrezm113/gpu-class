#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=sobel
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=sobel.%j.out

cd /scratch/$USER/GPUClass19/HOL5/sobel/basic/

set -o xtrace
./sobel ../input/fractal.pgm 100
./sobel ../input/world.pgm 100

echo "NVPROF"
nvprof ./sobel ../input/world.pgm 100

