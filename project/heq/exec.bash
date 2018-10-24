#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_heq
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec_heq.%j.out

module load legacy #To be able to load the old modules
module load opencv

cd /scratch/$USER/GPUClass18/FINPROJ/heq/

set -o xtrace
./heq input/bridge.png

