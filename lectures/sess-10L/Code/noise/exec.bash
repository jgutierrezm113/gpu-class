#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=noise
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=noise.%j.out

cd /scratch/$USER/GPUClass19/HOL5/noise/

set -o xtrace
./noise_remover ../input/5perSaPnoise.png
./measure_noise ./input.jpg
./measure_noise ./output_cpu.jpg
./measure_noise ./output_gpu.jpg

./noise_remover ../input/SaPnoise.png
./measure_noise ./input.jpg
./measure_noise ./output_cpu.jpg
./measure_noise ./output_gpu.jpg
