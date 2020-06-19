/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <cuda.h>

// Specified parameters for the Triple Spiral Valley 
double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
unsigned char* image_buffer_host;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
int colors[17][3] = {
                        {66, 30, 15},
                        {25, 7, 26},
                        {9, 1, 47},
                        {4, 4, 73},
                        {0, 7, 100},
                        {12, 44, 138},
                        {24, 82, 177},
                        {57, 125, 209},
                        {134, 181, 229},
                        {211, 236, 248},
                        {241, 233, 191},
                        {248, 201, 95},
                        {255, 170, 0},
                        {204, 128, 0},
                        {153, 87, 0},
                        {106, 52, 3},
                        {16, 16, 16},
};

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void compute_mandelbrot (unsigned char *image_buffer_device) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("meu i é %d || blockdim %d || blockid %d || threadid %d\n", i, blockDim.x, blockIdx.x, threadIdx.x);
    if (i < 10) image_buffer_device[i] = '4';
}

void allocate_image_buffer(unsigned char* image_buffer_device, int size) {
    // Our buffer, instead of a matrix, will be a big array

    // Allocate host memory
    image_buffer_host = (unsigned char*) malloc(sizeof(unsigned char) * size);

    // Allocate device memory
    image_buffer_device = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**) &image_buffer_device, size);

    // Test alloc success
    if (image_buffer_host == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
};

void init(int argc, char* argv[]) {
    if (argc < 6) {
        printf("usage: ./mandelbrot_seq c_x_min c_x_max c_y_min c_y_max image_size\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_seq -2.5 1.5 -2.0 2.0 11500\n");
        printf("    Seahorse Valley:      ./mandelbrot_seq -0.8 -0.7 0.05 0.15 11500\n");
        printf("    Elephant Valley:      ./mandelbrot_seq 0.175 0.375 -0.1 0.1 11500\n");
        printf("    Triple Spiral Valley: ./mandelbrot_seq -0.188 -0.012 0.554 0.754 11500\n");
        exit(0);
    }
    else {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);

        i_x_max = image_size;
        i_y_max = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width = (c_x_max - c_x_min) / i_x_max;
        pixel_height = (c_y_max - c_y_min) / i_y_max;
    };
};
/*
void update_rgb_buffer(int iteration, int x, int y) {
    int color;

    if (iteration == iteration_max) {
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};
*/
/*
void write_to_file() {
    FILE* file;
    char* filename = "output.ppm";
    char* comment = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
        i_x_max, i_y_max, max_color_component_value);

    for (int i = 0; i < image_buffer_size; i++) {
        fwrite(image_buffer[i], 1, 3, file);
    };

    fclose(file);
};
*/
int main(int argc, char *argv[]) {

    cudaError_t err = cudaSuccess;
    int rgb_size = 3;
    int size = image_buffer_size * rgb_size;

    init (argc, argv);

    unsigned char* image_buffer_device = NULL;
    allocate_image_buffer (image_buffer_device, size);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(image_size + threadsPerBlock - 1) / threadsPerBlock;
    compute_mandelbrot<<<1, 1 >>>(image_buffer_device);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch compute_mandelbrot kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(image_buffer_device, image_buffer_host, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    else {
        for (int i = 0; i < 10; i++) printf("buf[%d] = %uc\n", i, image_buffer_host[i]);
    }

    // Free device global memory
    err = cudaFree(image_buffer_device);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(image_buffer_host);

    return 0;
}

