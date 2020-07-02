OUTPUT=mandelbrot
OUT_DIR = bin
IMAGE=.ppm

MK_DIR = mkdir -p

NVCC=nvcc
MPI=mpicc
CC=gcc
CC_OPT=-std=c11

CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_cuda $(OUTPUT)_ompi

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(MK_DIR) $(OUT_DIR)
	$(NVCC) $(OUTPUT)_cuda.cu -o $(OUT_DIR)/$(OUTPUT)_cuda

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(MPI) $(OUTPUT)_ompi.c -g -o $(OUTPUT)_ompi

.PHONY: clean
clean:
	rm $(IMAGE) $(OUTPUT)_ompi
