OUTPUT=mandelbrot
OUT_DIR = bin
IMAGE=.ppm

MK_DIR = mkdir -p

NVCC=nvcc
MPI = mpicc
CC=gcc
CC_OPT=-std=c11

CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_cuda# $(OUTPUT)_pth $(OUTPUT)_seq

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(MK_DIR) $(OUT_DIR)
	$(NVCC) $(OUTPUT)_cuda.cu -o $(OUT_DIR)/$(OUTPUT)_cuda

# $(OUTPUT)_pth: $(OUTPUT)_pth.c
# 	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(OUTPUT)_pth.c

# $(OUTPUT)_seq: $(OUTPUT)_seq.c
# 	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

#$(OUTPUT)_mpi: $(OUTPUT)_mpi.c
#        $(MPI) -o $(OUTPUT)_mpi $(OUTPUT)_mpi.c

.PHONY: clean
clean:
	rm $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq *$(IMAGE) #$(OUTPUT)_mpi
