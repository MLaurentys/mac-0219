#! /bin/bash

set -o xtrace

MEASUREMENTS=15
SIZE=4096

# CUDA parameters
GRIDS=('4' '4' '16' '16' '64' '64' '256' '256' '1024' '1024' '2048' '2048')
GRIDS_NAME=('4_4' '16_16' '64_64' '256_256' '1024_1024' '2048_2048')
CUDA_BLOCKS=('1' '512' '1024' '2048')
THREADS=('4')
CUDA_RUN="./bin/mandelbrot_cuda"

# OMPI parameters
NUM_TASKS=('4' '8' '16' '32' '64')
MPI_RUN="mpirun -np"
MPI_BIN="./bin/mandelbrot_ompi"

MKDIR='mkdir -p'

#make
${MKDIR} results
${MKDIR} results/cuda
${MKDIR} results/ompi

# for ((i = 0; i < ${#GRIDS[@]}; i+=2)) do
#    perf stat -r $MEASUREMENTS $CUDA_RUN -0.188 -0.012 0.554 0.754 $SIZE "${GRIDS[$i]}"\
#        "${GRIDS[$i+1]}">> triple_spiral_${GRIDS_NAME[$i/2]}.log 2>&1
# done

for BLOCKS in ${CUDA_BLOCKS[@]}; do
    for ((THREADS=4; THREADS<=2048; THREADS*=2)); do
        perf stat -r $MEASUREMENTS $CUDA_RUN -0.188 -0.012 0.554 0.754 $SIZE $BLOCKS\
                     $THREADS >> triple_spiral_"$BLOCKS"_"$THREADS".log 2>&1
    done
done

mv *.log results/cuda/

for ((i = 0; i < ${#NUM_TASKS[@]}; i+=1)) do
    perf stat -r $MEASUREMENTS $MPI_RUN ${NUM_TASKS[$i]} $MPI_BIN -0.188 -0.012 0.554 0.754 $SIZE\
    >> triple_spiral_${NUM_TASKS[$i]}.log 2>&1
done
mv *.log results/ompi/

rm output.ppm