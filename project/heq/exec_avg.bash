#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_proj
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=exec_proj.%j.out

module load legacy #To be able to load the old modules
module load opencv

iterations=10

cd /scratch/$USER/GPUClass18/FINPROJ/heq/

#image
image="input/bridge.png"

echo "------------------------------"
echo "Running tests..."
echo "------------------------------"
echo "------------------------------"
echo ""

rm -rf run_exec.log

for ((i=1; i<= iterations; i++))
do
    echo "Running Trial $i"
    ./heq $image &>> run_exec.log
done

cpu_avg=`grep "CPU" run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
gpu_avg=`grep "GPU" run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
kernel_avg=`grep "Kernel" run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
diff_avg=`grep "Percentage" run_exec.log | awk '{ sum += $3; n++} END {if (n > 0) print sum/n}'`

echo "------------------------------"
echo "Results"
echo "    CPU Average: $cpu_avg"
echo "    GPU Average: $gpu_avg"
echo "    Ker Average: $kernel_avg"
echo "    Dif Average: $diff_avg"


