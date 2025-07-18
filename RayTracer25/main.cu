#include <iostream>

#include "lib/math_classes/vec3.h"
#include "lib/math_classes/color.h"

#include "lib/materials/material.h"

#include "lib/kernel_code.h"
#include "lib/random_cuda_funcs.h"
#include "lib/cuda_check_funcs.h"
#include "lib/camera.h"

#include "lib/scenes/base_test.h"
#include "lib/scenes/cornell_box.h"

#include <cuda_profiler_api.h>

// CUDA memory limits (TODO this will be important so I can check if I am using too much stack memory)
#define heap_size (3221225472) // 3 GB heap size (1/4 of my GPU memory)
// #define stack_size (536870912) // 536870912 is 0.5 GB stack size //3221225472 is 3 GB stack size (1/4 of my GPU memory)
#define stack_size (65536) // 536870912 is 0.5 GB stack size //3221225472 is 3 GB stack size (1/4 of my GPU memory)

// Framebuffer dimensions and rays per pixel
// 8k resolution is 7680x4320, but I will use 1024x1024 for testing because 8k is too large to view in vscode ppm extension (id have to use infranviewer)
#define img_width 1024
#define img_height 1024   
// #define img_width 2048
// #define img_height 2048   
#define rays_per_pixel 3000
#define hittable_max 10

#define max_bounce_count 20 // Maximum number of bounces for ray tracing



int main() {
    std::cout << "Heap Limit Set to " << heap_size << std::endl;
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    checkCudaErrors(cudaGetLastError());


    std::cout << "Stack Limit Set to " << stack_size << std::endl;
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
    checkCudaErrors(cudaGetLastError());


    // define blocks and threads
    std::cout << "Block and Thread Size Defined" << std::endl;
    dim3 blocks((img_width + 15) / 16, (img_height + 15) / 16);
    dim3 threads(16, 16);
    checkCudaErrors(cudaGetLastError());


    // -------------------------------------------------------------
    // -------------------- setup random seeds --------------------- 
    // -------------------------------------------------------------
    int seed = 1234; // Example seed, TODO change this to maybe a time based seed?
    curandState *device_rand_states;
    checkCudaErrors(cudaMalloc((void**)&device_rand_states, img_width * img_height * sizeof(curandState)));
    kernel_init_curand_states<<<blocks, threads>>>(device_rand_states, img_width, img_height, seed);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());


    // -------------------------------------------------------------
    // --------------- setup camera & hittables --------------------    
    // -------------------------------------------------------------
    std::cout << "Building Scene" << std::endl;
    
    camera *device_cam;
    hittable *device_hittables;
    int hittable_count = 0; // Number of hittable objects
    // int scene_err = make_base_scene(device_cam, device_hittables, hittable_count, img_width, img_height, rays_per_pixel, max_bounce_count, hittable_max);
    int scene_err = make_cornell_box_scene(device_cam, device_hittables, hittable_count, img_width, img_height, rays_per_pixel, max_bounce_count, hittable_max);
    checkCudaErrors(cudaGetLastError());


    if(scene_err) return -1;
    std::cout << "Scene Built" << std::endl;

    // -------------------------------------------------------------
    // ----------------- define the framebuffer --------------------
    // -------------------------------------------------------------
    uint8_t *framebuffer = (uint8_t*)malloc(img_height * img_width * 3 * sizeof(uint8_t));
    if (!framebuffer) {
        std::cerr << "Failed to allocate framebuffer memory!" << std::endl;
        return -1;
    }
    for (int i = 0; i < img_height * img_width * 3; ++i) {
        framebuffer[i] = 0; // Initialize framebuffer to zero
    }

    // Allocate device memory for framebuffer 
    uint8_t* device_framebuffer; 
    checkCudaErrors(cudaMalloc((void**)&device_framebuffer, img_height * img_width * 3 * sizeof(uint8_t)));
    checkCudaErrors(cudaMemcpy(device_framebuffer, framebuffer, img_height * img_width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // -------------------------------------------------------------
    // ---------------------- Launch the kernel --------------------
    // -------------------------------------------------------------
    cudaProfilerStart();
    std::cout << "launching Kernel" << std::endl;
    kernel<<<blocks, threads>>>(device_framebuffer, device_cam, device_hittables, hittable_count, device_rand_states);
    std::cout << "Kernel returned" << std::endl;
    cudaProfilerStop();
    checkCudaErrors(cudaGetLastError());


    // -------------------------------------------------------------
    // ------------------- Synchronize the device ------------------
    // -------------------------------------------------------------
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    // Copy the framebuffer back to host memory
    cudaMemcpy(framebuffer, device_framebuffer, img_height * img_width * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_framebuffer);


    // Output the framebuffer to a ppm file
    FILE *file = fopen("output.ppm", "wb");
    fprintf(file, "P6 %d %d 255\n", img_width, img_height);
    fwrite(framebuffer, sizeof(uint8_t), img_height * img_width * 3, file);
    fclose(file);

    free (framebuffer);


    // TODO free cuda memory

    return 0;
}