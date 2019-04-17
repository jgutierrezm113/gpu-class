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

semester="GPUClassS19"
iterations=10

cd /scratch/$USER/$semester/PROJEVAL/heq/

echo "------------------------------"
echo "Running tests..."
echo "------------------------------"
echo "------------------------------"
echo ""

input_data=( \
    /scratch/gutierrez.jul/$semester/FINPROJ/heq/src/  \
    /scratch/dong.yiw/$semester/FINPROJ/heq/src/  \
    /scratch/sharma.him/FINPROJ/heq/src/  \
    /scratch/palmer.b/$semester/FINPROJ/heq/src/  \
    /scratch/lee.kuan/$semester/FINPROJ/heq/src/  \
    /scratch/zandigohar.m/FINPROJ/heq/src/  \
    /scratch/becker-wheeler.k/GPUFINALS19/src/ \
    /scratch/akella.p/$semester/FINPROJ/heq/src/ \
    /scratch/gaggar.s/gpuclass/FINPROJ/heq/src/ \

    )

#get folder recollection
MYPWD=${PWD}

input_images="img0 img1 img2 img3"

#NOTICE REMOVING RESULTS!!
rm -rf results/

mkdir -p $MYPWD/results/

echo "user,img0,,img1,,img2,,img3,,total time" > $MYPWD/results/summary.csv
echo ",time,diff,time,diff,time,diff,time,diff" >> $MYPWD/results/summary.csv

for data in ${input_data[@]}
do
    if [ -f $data/heq.cu ]; then
        #figure out username
        username=`echo $data | sed "s/\/scratch\/\([a-zA-Z0-9.]\+\).*/\1/"`
        mkdir -p $MYPWD/results/$username/

        echo "------------------------------"
        echo "User: $username"
        echo "------------------------------"
        
        #copy file into current folder
        cp $data/heq.cu src/
        cp $data/heq.cu $MYPWD/results/$username/
        
        #compile code 
        echo "Running Make"
        
        make all
        
        total_avg=0.0
        echo -n "$username," >> $MYPWD/results/summary.csv
        for input in $input_images
        do
            echo ""
            echo "------------------------------"
            echo " Input $input"
            echo "------------------------------"
            
            echo -n "Running Trial: "
            for ((i=1; i<= iterations; i++))
            do
                echo -n "$i,"
                ./heq input/${input}.jpg &>> $MYPWD/results/$username/${input}-exec.log
                mv input.jpg $MYPWD/results/${input}.jpg
                mv output_cpu.jpg $MYPWD/results/$username/${input}-cpu-out.jpg
                mv output_gpu.jpg $MYPWD/results/$username/${input}-gpu-out.jpg
            done
            
            gpu_avg=`grep "GPU"        results/$username/${input}-exec.log | awk '{ sum += $4; n++} END {if (n > 0) print sum/n}'`
            dif_avg=`grep "Percentage" results/$username/${input}-exec.log | awk '{ sum += $3; n++} END {if (n > 0) print sum/n}'`
            
            echo ""
            echo "------------------------------"
            echo "Results"
            echo "    GPU Average: $gpu_avg"
            echo "    Dif Average: $dif_avg"
            
            echo -n "$gpu_avg,$dif_avg," >> $MYPWD/results/summary.csv
            total_avg=`awk "BEGIN {print $gpu_avg + $total_avg; exit}"`
        done
        
        echo "$total_avg," >> $MYPWD/results/summary.csv
        rm src/heq.cu
        make clean
    fi
done

cd $MYPWD

