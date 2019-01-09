#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof_coalesced
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof_coalesced.%j.out

cd /scratch/$USER/GPUClass18/HOL2/

set -o xtrace
nvprof --metrics gld_requested_throughput, \
                 gst_requested_throughput, \
                 gst_throughput, \
                 gld_throughput, \
                 gld_efficiency, \
                 gst_efficiency, \
                 stall_memory_dependency, \
                 gld_transactions_per_request, \
                 gst_transaction_per_request \
                 ./vadd_coalesced 100000000 

