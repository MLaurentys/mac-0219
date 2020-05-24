#! /bin/bash

set -o xtrace

MEASUREMENTS=13
ITERATIONS=1
INITIAL_SIZE=32
TH_NUM=('32')
SIZE=$INITIAL_SIZE

NAMES_TH=('mandelbrot_pth' 'mandelbrot_omp') 
make

for NAME in ${NAMES_TH[@]}; do
    mkdir results/$NAME
    
    for TH in ${TH_NUM[@]}; do

    	for ((i=1; i<=$ITERATIONS; i++)); do
           	 perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE $TH>> full$TH.log 2>&1
	   	 perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE $TH>> seahorse$TH.log 2>&1
       		 perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE $TH>> elephant$TH.log 2>&1
	         perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE $TH>> triple_spiral$TH.log 2>&1
           	 SIZE=$(($SIZE * 2))
    	 done

         SIZE=$INITIAL_SIZE

    done
done
rm output.ppm
