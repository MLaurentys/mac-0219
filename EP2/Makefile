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
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_ompi_omp

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(MK_DIR) $(OUT_DIR)
	$(NVCC) $(OUTPUT)_cuda.cu -o $(OUT_DIR)/$(OUTPUT)_cuda

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(MPI) $(OUTPUT)_ompi.c -g -o $(OUT_DIR)/$(OUTPUT)_ompi

$(OUTPUT)_ompi_omp: $(OUTPUT)_ompi_omp.c
	$(MPI) $(OUTPUT)_ompi_omp $(CC_OMP) -o $(OUT_DIR)/$(OUTPUT)_ompi_omp

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUT_DIR)/$(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(OUTPUT)_omp.c

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUT_DIR)/$(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(OUTPUT)_pth.c

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUT_DIR)/$(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c
