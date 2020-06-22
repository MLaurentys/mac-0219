#! /bin/bash

set -o xtrace

MEASUREMENTS=1
ITERATIONS=1
INITIAL_SIZE=16
GRIDS=('4' '4' '16' '16' '64' '64' '256' '256' '1024' '1024')
GRIDS_NAME=("4_4" "16_16" '64_64' '256_256' '1024_1024')
INITIAL_SIZE=16
MDIR=mkdir -p
CUDA_RUN="./bin/mandelbrot_cuda"

make
${MDIR} results
${MDIR} results/cuda
${MDIR} results/ompi

SIZE=$INITIAL_SIZE
NAME='mandelbrot_cuda'
echo ${GRIDS}
for ((j = 0; j <= $ITERATIONS; j++))do
    for ((i = 0; i < ${#GRIDS[@]}; i+=2)) do
        perf stat -r $MEASUREMENTS $CUDA_RUN -2.5 1.5 -2.0 2.0 $SIZE "${GRIDS[$i]}"\
            "${GRIDS[$i+1]}">> full_${GRIDS_NAME[$i/2]}.log 2>&1
        perf stat -r $MEASUREMENTS $CUDA_RUN -2.5 1.5 -2.0 2.0 $SIZE "${GRIDS[$i]}"\
            "${GRIDS[$i+1]}">> seahorse_${GRIDS_NAME[$i/2]}.log 2>&1
        perf stat -r $MEASUREMENTS $CUDA_RUN -2.5 1.5 -2.0 2.0 $SIZE "${GRIDS[$i]}"\
            "${GRIDS[$i+1]}">> elephant_${GRIDS_NAME[$i/2]}.log 2>&1
        perf stat -r $MEASUREMENTS $CUDA_RUN -2.5 1.5 -2.0 2.0 $SIZE "${GRIDS[$i]}"\
            "${GRIDS[$i+1]}">> triple_spiral_${GRIDS_NAME[$i/2]}.log 2>&1
    done
    SIZE=$(($SIZE * 2))
done
SIZE=$INITIAL_SIZE

mv *.log results/cuda/
rm output.ppm
