#! /bin/bash

set -o xtrace

MEASUREMENTS=15
SIZE=4096
GRIDS=('4' '4' '16' '16' '64' '64' '256' '256' '1024' '1024' '2048' '2048')
GRIDS_NAME=('4_4' '16_16' '64_64' '256_256' '1024_1024' '2048_2048')
MDIR=mkdir -p
CUDA_RUN="./bin/mandelbrot_cuda"

make
${MKDIR} results
${MKDIR} results/cuda
${MKDIR} results/ompi

echo ${GRIDS}
for ((i = 0; i < ${#GRIDS[@]}; i+=2)) do
    perf stat -r $MEASUREMENTS $CUDA_RUN -0.188 -0.012 0.554 0.754 $SIZE "${GRIDS[$i]}"\
        "${GRIDS[$i+1]}">> triple_spiral_${GRIDS_NAME[$i/2]}.log 2>&1
done

mv *.log results/cuda/
rm output.ppm