#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=exec_proj
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=exec_proj.%j.out

module load legacy #To be able to load the old modules
module load opencv

cd /scratch/$USER/GPUClass18/PROJEVAL/heq/
iterations=10

echo "------------------------------"
echo "Running tests..."
echo "------------------------------"
echo "------------------------------"
echo ""

input_data=( \
    /scratch/gutierrez.jul/GPUClass18/FINPROJ/heq/src/ \
    /scratch/gutierrez.jul/GPUClass18/FINPROJ2/heq/src/ \
    )

#unkown image
unknown_image="input/low-contrast.jpg"

#get folder recollection
MYPWD=${PWD}

#NOTICE REMOVING RESULTS!!
rm -rf results/

mkdir -p $MYPWD/results/

echo "user, cpu, gpu, kernel, difference" > $MYPWD/results/summary.csv

for data in ${input_data[@]}
do
    #figure out username
    username=`echo $data | sed "s/\/scratch\/\([a-zA-Z0-9.]\+\).*/\1/"`
	mkdir -p $MYPWD/results/$username/

    #copy file into current folder
    cp $data/heq.cu src/
    cp $data/heq.cu $MYPWD/results/$username/
    
    #compile code 
    echo "------------------------------"
    echo "User: $username"
    echo "------------------------------"
    echo "Running Make"
    
    make
    echo ""
    echo "------------------------------"
    
    for ((i=1; i<= iterations; i++))
    do
        echo "Running Trial $i"
        ./heq $unknown_image $MYPWD/ &>> $MYPWD/results/$username/run_exec.log
        mv input_baw.jpg $MYPWD/results/
        mv output_cpu.jpg $MYPWD/results/$username/
        mv output_gpu.jpg $MYPWD/results/$username/
    done
    
    cpu_avg=`grep "CPU" results/$username/run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
    gpu_avg=`grep "GPU" results/$username/run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
    kernel_avg=`grep "Kernel" results/$username/run_exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
    diff_avg=`grep "Percentage" results/$username/run_exec.log | awk '{ sum += $3; n++} END {if (n > 0) print sum/n}'`
    
    echo "------------------------------"
    echo "Results"
    echo "    CPU Average: $cpu_avg"
    echo "    GPU Average: $gpu_avg"
    echo "    Ker Average: $kernel_avg"
    echo "    Dif Average: $diff_avg"
    
    echo "$username, $cpu_avg, $gpu_avg, $kernel_avg, $diff_avg" >> $MYPWD/results/summary.csv

done

cd $MYPWD

