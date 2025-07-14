#include <iostream>

#include "lib/math_classes/vec3.h"
#include "lib/math_classes/color.h"

#include "lib/materials/material.h"

#include "lib/kernel_code.h"
#include "lib/random_cuda_funcs.h"

// CUDA memory limits (TODO this will be important so I can check if I am using too much stack memory)
#define heap_size (3221225472) // 3 GB heap size (1/4 of my GPU memory)
#define stack_size (536870912) // 536870912 is 0.5 GB stack size //3221225472 is 3 GB stack size (1/4 of my GPU memory)

// Framebuffer dimensions and rays per pixel
// 8k resolution is 7680x4320, but I will use 1024x1024 for testing because 8k is too large to view in vscode ppm extension (id have to use infranviewer)
#define img_width 1024
#define img_height 1024   
#define rays_per_pixel 500
#define hittable_max 100
#define max_bounce_count 50 // Maximum number of bounces for ray tracing


#include "lib/cuda_check_funcs.h"

int main() {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size);


    std::cout << "Hello, CUDA World!" << std::endl;

    // define blocks and threads
    dim3 blocks((img_width + 15) / 16, (img_height + 15) / 16);
    dim3 threads(16, 16);

    // -------------------------------------------------------------
    // -------------------- setup random seeds --------------------- 
    // -------------------------------------------------------------
    int seed = 1234; // Example seed, TODO change this to maybe a time based seed?
    curandState *device_rand_states;
    checkCudaErrors(cudaMalloc((void**)&device_rand_states, img_width * img_height * sizeof(curandState)));
    kernel_init_curand_states<<<blocks, threads>>>(device_rand_states, img_width, img_height, seed);
    checkCudaErrors(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // ---------------------- define camera ------------------------
    // -------------------------------------------------------------
    camera *cam = (camera*)malloc(sizeof(camera));
    if (!cam) {
        std::cerr << "Failed to allocate camera memory!" << std::endl;
        return -1;
    }
    cam->cam_width = img_width;
    cam->cam_height = img_height;
    cam->origin = new_vec3(0, 0, 0); // camera position
    cam->focal_length = 2.0f; // focal length of the camera
    cam->viewport_height = 2.0f; // default viewport height
    cam->viewport_width = (float)img_width / (float)img_height * cam->viewport_height; // calculate viewport width based on aspect ratio
    cam->samples_per_pixel = rays_per_pixel; // number of samples per pixel for anti-aliasing
    cam->max_bounces = max_bounce_count; // maximum number of bounces for ray tracing

    // Allocate device memory for camera
    camera *device_cam;
    checkCudaErrors(cudaMalloc((void**)&device_cam, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(device_cam, cam, sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(cam); // Free host memory for camera after copying to device

    // -------------------------------------------------------------
    // -------------------- define hittables -----------------------
    // -------------------------------------------------------------
    material lambertian_red = new_material_lambertian(new_color(1.0f, 0.0f, 0.0f)); // Example material
    material lambertian_green = new_material_lambertian(new_color(0.0f, 1.0f, 0.0f)); // Example material
    hittable *hittables = (hittable*)malloc(hittable_max * sizeof(hittable));
    int hittable_count = 0; // Number of hittable objects, for now just one sphere
    if (!hittables) {
        std::cerr << "Failed to allocate hittables memory!" << std::endl;
        return -1;
    }
    hittables[hittable_count] = new_hittable_sphere(new_vec3(0, 0, 2), 0.5f, lambertian_red); // Example sphere
    hittable_count+= 1; // Set hittable count to 1 for now
    hittables[hittable_count] = new_hittable_sphere(new_vec3(0, -500, 2), 499.0f, lambertian_green); // Example ground sphere
    hittable_count+= 1; // Increment hittable count

    // Allocate device memory for hittables
    hittable *device_hittables;
    checkCudaErrors(cudaMalloc((void**)&device_hittables, hittable_count * sizeof(hittable)));
    checkCudaErrors(cudaMemcpy(device_hittables, hittables, hittable_count * sizeof(hittable), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(hittables); // Free host memory for hittables after copying to device

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

    // -------------------------------------------------------------
    // ---------------------- Launch the kernel --------------------
    // -------------------------------------------------------------
    kernel<<<blocks, threads>>>(device_framebuffer, device_cam, device_hittables, hittable_count, device_rand_states);
    // checkCudaErrors(cudaGetLastError());


    // -------------------------------------------------------------
    // ------------------- Synchronize the device ------------------
    // -------------------------------------------------------------
    checkCudaErrors(cudaDeviceSynchronize());
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

    return 0;
}