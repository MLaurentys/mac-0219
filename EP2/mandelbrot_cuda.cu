
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

int num_blocks;
int th_per_block;
__device__ dvc_num_blocks;


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

void init(int argc, char* argv[]) {
    if (argc < 8) {
        printf("usage: ./mandelbrot_seq c_x_min c_x_max c_y_min c_y_max"
            " image_size NUM_BLOCKS TH_PER_BLOCK\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_seq -2.5 1.5 -2.0 2.0 11500 4 64\n");
        printf("    Seahorse Valley:      ./mandelbrot_seq -0.8 -0.7 0.05 0.15 11500 4 64\n");
        printf("    Elephant Valley:      ./mandelbrot_seq 0.175 0.375 -0.1 0.1 11500 4 64\n");
        printf("    Triple Spiral Valley: ./mandelbrot_seq -0.188 -0.012 0.554 0.754 11500 4 64\n");
        exit(0);
    }
    else {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &num_blocks);
        sscanf(argv[7], "%d", &th_per_block

            i_x_max = image_size;
        i_y_max = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width = (c_x_max - c_x_min) / i_x_max;
        pixel_height = (c_y_max - c_y_min) / i_y_max;
    };
};

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__
void compute_mandelbrot(unsigned char* image_buffer_device, int buffer_size, int row_size) {
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i_x;
    int i_y;

    double c_x;
    double c_y;

    // Example: image 5x5 -> buffer_size = 25
    // 3 blocks of 3 threads -> 9 threads
    // 
    // what thread will process each pixel:
    // 2 4 7 - -
    // 1 4 6 - -
    // 1 3 6 8 -
    // 0 3 5 8 -
    // 0 2 5 7 -
    // 
    // 1 3 4 6 8
    // 1 2 4 6 7
    // 0 2 4 5 7
    // 0 2 3 5 7
    // 0 1 3 5 6
    //
    // and the remaining pixels we process separetedly,
    // each thread process some remaining pixels in the end

    int pixels_per_thread = size / (blockDim.x * #numeroDeBlocos)
    int my_thread = blockDim.x * blockIdx.x + threadIdx.x;

    // Its easier to process by pixels instead of by row-collunm
    int init_pixel = my_thread * pixels_per_thread;
    int end_pixel = init_pixel + pixels_per_thread;

    for (int pix = init_pixel; pix < end_pixel; pix++) {

        i_y = pix / row_size;
        i_x = pix % row_size;

        c_y = c_y_min + i_y * pixel_height;
        if (fabs(c_y) < pixel_height / 2) {
            c_y = 0.0;
        };

        c_x = c_x_min + i_x * pixel_width;

        z_x = 0.0;
        z_y = 0.0;

        z_x_squared = 0.0;
        z_y_squared = 0.0;

        for (iteration = 0;
            iteration < iteration_max && \
            ((z_x_squared + z_y_squared) < escape_radius_squared);
            iteration++) {
            z_y = 2 * z_x * z_y + c_y;
            z_x = z_x_squared - z_y_squared + c_x;

            z_x_squared = z_x * z_x;
            z_y_squared = z_y * z_y;
        };
        update_rgb_buffer(image_buffer_device, iteration, i_x, i_y);
    }
    // Remaining pixels
    int num_threads = blockDim.x * #numeroDeBlocos;
    int first_remaining_pixel = buffer_size - (num_threads * pixels_per_thread);
    int rem_pixel_per_thread = (buffer_size - first_remaining_pixel)/num_threads + 1;
    int my_first_rem = first_remaining_pixel + rem_pixel_per_thread * my_thread;
    int my_last_rem = my_rem_pix + rem_pixel_per_thread;
    // For dos pixels remanescente para fazer ainda

}

void allocate_image_buffer(unsigned char** image_buffer_device, size_t size) {
    // Our buffer, instead of a matrix, will be a big array

    // Allocate host memory
    image_buffer_host = (unsigned char*)malloc(sizeof(unsigned char) * size);

    // Allocate device memory
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)(image_buffer_device), size);

    // Test alloc success
    if (*image_buffer_host == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
};



void update_rgb_buffer(unsigned char* image_buffer_device, int iteration, int x, int y) {
    int color;

    if (iteration == iteration_max) {
        image_buffer_device[(i_y_max * y) + x + 0] = colors[gradient_size][0];
        image_buffer_device[(i_y_max * y) + x + 1] = colors[gradient_size][1];
        image_buffer_device[(i_y_max * y) + x + 2] = colors[gradient_size][2];
    }
    else {
        color = iteration % gradient_size;

        image_buffer_device[(i_y_max * y) + x + 0] = colors[color][0];
        image_buffer_device[(i_y_max * y) + x + 1] = colors[color][1];
        image_buffer_device[(i_y_max * y) + x + 2] = colors[color][2];
    };
};

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

int main(int argc, char* argv[]) {
    cudaError_t err;
    
    init(argc, argv);

    int rgb_size = 3;
    size_t size = image_buffer_size * rgb_size;
    err = cudaMemcpyToSymbol(dvc_num_blocks, &num_blocks, sizeof(int));
    if (err != cudaSuccess) {
        printf("Failed to set value to device variable (error code"
        " %s)!\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    unsigned char* image_buffer_device;
    allocate_image_buffer(&image_buffer_device, size);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (image_size + threadsPerBlock - 1) / threadsPerBlock;
    compute_mandelbrot << <num_blocks, th_per_block >> > (
        image_buffer_device, image_buffer_size, image_size);
    cudaDeviceSynchronize();
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch compute_mandelbrot kernel"
            " (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(image_buffer_host, image_buffer_device, size,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector from device to host"
            " (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 10; i++)
        printf("host[%d] = %u\n", i, image_buffer_host[i]);

    // Free device global memory
    err = cudaFree(image_buffer_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector (error code"
            " %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free host memory
    free(image_buffer_host);

    return 0;
}