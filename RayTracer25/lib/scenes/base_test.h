#ifndef BASE_TEST_H
#define BASE_TEST_H

#include "../hittable/hittable.h"
#include "../camera.h"

#include <iostream>
#include "../cuda_check_funcs.h"


__host__ int make_base_scene(camera *&device_camera, hittable *&device_hittables, int& hittable_count, int img_width, int img_height, int rays_per_pixel, int max_bounce_count, int hittable_max) {
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
    cam->origin = new_vec3(-2, 1, -1); // camera position
    cam->focal_length = 2.0f; // focal length of the camera
    cam->viewport_height = 2.0f; // default viewport height
    cam->viewport_width = (float)img_width / (float)img_height * cam->viewport_height; // calculate viewport width based on aspect ratio
    cam->samples_per_pixel = rays_per_pixel; // number of samples per pixel for anti-aliasing
    cam->max_bounces = max_bounce_count; // maximum number of bounces for ray tracing
    cam->look_at_pos = new_vec3(0, 0, 2); // Look at position
    camera_calc_look_at(cam); // Calculate camera orientation vectors

    // Allocate device memory for camera
    checkCudaErrors(cudaMalloc((void**)&device_camera, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(device_camera, cam, sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(cam); // Free host memory for camera after copying to device

    // -------------------------------------------------------------
    // -------------------- define hittables -----------------------
    // -------------------------------------------------------------
    material lambertian_red = new_material_lambertian(new_color(1.0f, 0.0f, 0.0f)); // Example material
    material lambertian_yellow = new_material_lambertian(new_color(1.0f, 1.0f, 0.0f)); // Example material
    material lambertian_green = new_material_lambertian(new_color(0.0f, 1.0f, 0.0f)); // Example material

    material light = new_material_emissive(new_color(1.0f, 1.0f, 1.0f), 10.0f); // Example emissive material

    // material mirror_metal = new_material_metal(new_color(0.8f, 0.8f, 0.8f), 0.0f); // Example metal material
    material glass_dielectric = new_material_dielectric(1.5f); // Example dielectric material

    hittable *hittables = (hittable*)malloc(hittable_max * sizeof(hittable));
    if (!hittables) {
        std::cerr << "Failed to allocate hittables memory!" << std::endl;
        return -1;
    }
    hittables[hittable_count++] = new_hittable_sphere(new_vec3(0, 0, 2), 0.5, light); // Example sphere
    hittables[hittable_count++] = new_hittable_sphere(new_vec3(0.7, -0.4, 1.25), 0.1, lambertian_red); // Example sphere
    hittables[hittable_count++] = new_hittable_sphere(new_vec3(0, 0, 5), 0.1, lambertian_yellow); // Example sphere
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(-10, -1,-10), 
                                                        new_vec3( 10, -1, 10), 
                                                        new_vec3( 10, -1,-10), 
                                                        lambertian_green); // Example ground polygon
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(-10, -1,-10), 
                                                        new_vec3(-10, -1, 10), 
                                                        new_vec3( 10, -1, 10), 
                                                        lambertian_green); // Example ground polygon


    // Allocate device memory for hittables
    checkCudaErrors(cudaMalloc((void**)&device_hittables, hittable_count * sizeof(hittable)));
    checkCudaErrors(cudaMemcpy(device_hittables, hittables, hittable_count * sizeof(hittable), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(hittables); // Free host memory for hittables after copying to device

    return 0;
}

#endif // BASE_TEST_H