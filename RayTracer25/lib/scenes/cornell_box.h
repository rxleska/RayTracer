#ifndef CORNELL_BOX_H
#define CORNELL_BOX_H

#include "../hittable/hittable.h"
#include "../camera.h"

#include <iostream>
#include "../cuda_check_funcs.h"

__host__ int make_cornell_box_scene(camera *&device_camera, hittable *&device_hittables, int& hittable_count, int img_width, int img_height, int rays_per_pixel, int max_bounce_count, int hittable_max) {
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
    cam->origin = new_vec3(278.0f, 278.0f, -400.0f); 
    cam->focal_length = 15.0f;
    cam->viewport_height = 1.5f; 
    cam->viewport_width = (float)img_width / (float)img_height * cam->viewport_height; 
    cam->samples_per_pixel = rays_per_pixel;
    cam->max_bounces = max_bounce_count; 
    cam->look_at_pos = new_vec3(278.0f, 278.0f, 0.0f); 
    camera_calc_look_at(cam); 

    // Allocate device memory for camera
    checkCudaErrors(cudaMalloc((void**)&device_camera, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(device_camera, cam, sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(cam); // Free host memory for camera after copying to device

    // -------------------------------------------------------------
    // -------------------- define hittables -----------------------
    // -------------------------------------------------------------
    material white  = new_material_lambertian(new_color(1.0,1.0,1.0));
    material light  = new_material_emissive(new_color(1.0,1.0,1.0), 10.0f);
    material green  = new_material_lambertian(new_color(0.12,0.45,0.15));
    material red    = new_material_lambertian(new_color(0.65,0.05,0.05));
    material pink    = new_material_lambertian(new_color(1.0,0,1.0));
    // material mirror = new_material_metal(new_color(0.9,0.9,0.9),0.001f);

    hittable *hittables = (hittable*)malloc(hittable_max * sizeof(hittable));
    if (!hittables) {
        std::cerr << "Failed to allocate hittables memory!" << std::endl;
        return -1;
    }

    // floor
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(552.8, 0.0, 0.0),new_vec3(0.0, 0.0, 0.0),new_vec3(0.0, 0.0, 559.2), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(552.8, 0.0, 0.0),new_vec3(0.0, 0.0, 559.2),new_vec3(552.8, 0.0, 559.2), white);

    //ceiling light 
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(343.0, 548.5, 227.0),new_vec3(343.0, 548.5, 332.0),new_vec3(213.0, 548.5, 332.0), light);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(343.0, 548.5, 227.0),new_vec3(213.0, 548.5, 332.0),new_vec3(213.0, 548.5, 227.0), light);

    //Ceiling
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(556.0, 548.8, 0.0),new_vec3(556.0, 548.8, 559.2),new_vec3(0.0, 548.8, 559.2), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(556.0, 548.8, 0.0),new_vec3(0.0, 548.8, 559.2),new_vec3(0.0, 548.8, 0.0), white);

    //back wall
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(549.6, 0.0, 559.2),new_vec3(0.0, 0.0, 559.2),new_vec3(0.0, 548.8, 559.2), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(549.6, 0.0, 559.2),new_vec3(0.0, 548.8, 559.2),new_vec3(556.0, 548.8, 559.2), white);

    //right wall
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(0.0, 0.0, 559.2),new_vec3(0.0, 0.0, 0.0),new_vec3(0.0, 548.8, 0.0), green);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(0.0, 0.0, 559.2),new_vec3(0.0, 548.8, 0.0),new_vec3(0.0, 548.8, 559.2), green);

    //left wall
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(552.8, 0.0, 0.0),new_vec3(549.6, 0.0, 559.2),new_vec3(556.0, 548.8, 559.2), red);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(552.8, 0.0, 0.0),new_vec3(556.0, 548.8, 559.2),new_vec3(556.0, 548.8, 0.0), red);

    // //camera wall (we can see through this due to the directionality of Polygon_Ts)
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(549.6, 0.0, 0.0),new_vec3(0.0, 548.8, 0.0),new_vec3(0.0, 0.0, 0.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(549.6, 0.0, 0.0),new_vec3(556.0, 548.8, 0.0),new_vec3(0.0, 548.8, 0.0), white);

    //short block
    //wall1
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(130.0, 165.0, 65.0),
                                                        new_vec3(82.0, 165.0, 225.0),
                                                        new_vec3(240.0, 165.0, 272.0), 
                                                        white);
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(130.0, 165.0, 65.0),
                                                        new_vec3(240.0, 165.0, 272.0),
                                                        new_vec3(290.0, 165.0, 114.0), 
                                                        white);
    
    
    //wall2
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(290.0, 0.0, 114.0),
                                                        new_vec3(290.0, 165.0, 114.0),
                                                        new_vec3(240.0, 165.0, 272.0), 
                                                        white);
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(290.0, 0.0, 114.0),
                                                        new_vec3(240.0, 165.0, 272.0),
                                                        new_vec3(240.0, 0.0, 272.0), 
                                                        white);
    
    
    //wall3
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(130.0, 0.0, 65.0),
                                                        new_vec3(130.0, 165.0, 65.0),
                                                        new_vec3(290.0, 165.0, 114.0), 
                                                        white);
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(130.0, 0.0, 65.0),
                                                        new_vec3(290.0, 165.0, 114.0),
                                                        new_vec3(290.0, 0.0, 114.0), 
                                                        white);
    
    
    //wall4
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(82.0, 0.0, 225.0),
                                                        new_vec3(82.0, 165.0, 225.0),
                                                        new_vec3(130.0, 165.0, 65.0), 
                                                        white);
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(82.0, 0.0, 225.0),
                                                        new_vec3(130.0, 165.0, 65.0),
                                                        new_vec3(130.0, 0.0, 65.0), 
                                                        white);
    
    
    //wall5
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(240.0, 0.0, 272.0),
                                                        new_vec3(240.0, 165.0, 272.0),
                                                        new_vec3(82.0, 165.0, 225.0), 
                                                        white);
    hittables[hittable_count++] = new_hittable_polygon( new_vec3(240.0, 0.0, 272.0),
                                                        new_vec3(82.0, 165.0, 225.0),
                                                        new_vec3(82.0, 0.0, 225.0), 
                                                        white);
    

    //tall block
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(423.0, 330.0, 247.0),new_vec3(265.0, 330.0, 296.0),new_vec3(314.0, 330.0, 456.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(423.0, 330.0, 247.0),new_vec3(314.0, 330.0, 456.0),new_vec3(472.0, 330.0, 406.0), white);
    
    
    //wall2
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(423.0, 0.0, 247.0),new_vec3(423.0, 330.0, 247.0),new_vec3(472.0, 330.0, 406.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(423.0, 0.0, 247.0),new_vec3(472.0, 330.0, 406.0),new_vec3(472.0, 0.0, 406.0), white);
    
    //wall3
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(472.0, 0.0, 406.0),new_vec3(472.0, 330.0, 406.0),new_vec3(314.0, 330.0, 456.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(472.0, 0.0, 406.0),new_vec3(314.0, 330.0, 456.0),new_vec3(314.0, 0.0, 456.0), white);
    
    
    //wall4
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(314.0, 0.0, 456.0),new_vec3(314.0, 330.0, 456.0),new_vec3(265.0, 330.0, 296.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(314.0, 0.0, 456.0),new_vec3(265.0, 330.0, 296.0),new_vec3(265.0, 0.0, 296.0), white);
    
    //wall5
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(265.0, 0.0, 296.0),new_vec3(265.0, 330.0, 296.0),new_vec3(423.0, 330.0, 247.0), white);
    hittables[hittable_count++] = new_hittable_polygon(new_vec3(265.0, 0.0, 296.0),new_vec3(423.0, 330.0, 247.0),new_vec3(423.0, 0.0, 247.0), white);


    // Allocate device memory for hittables
    checkCudaErrors(cudaMalloc((void**)&device_hittables, hittable_count * sizeof(hittable)));
    checkCudaErrors(cudaMemcpy(device_hittables, hittables, hittable_count * sizeof(hittable), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    free(hittables); // Free host memory for hittables after copying to device
    return 0;
}

#endif // CORNELL_BOX_H