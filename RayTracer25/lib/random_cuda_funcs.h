#ifndef RANDOM_CUDA_FUNCS_H
#define RANDOM_CUDA_FUNCS_H

#include <curand_kernel.h>

__global__ void kernel_init_curand_states(curandState *states, int width, int height, unsigned long seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int pixel_index = y * width + x;
    curand_init(seed, pixel_index, 0, &states[pixel_index]);
}


#endif // RANDOM_CUDA_FUNCS_H