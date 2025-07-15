#ifndef KERNEL_CODE_H
#define KERNEL_CODE_H

#include "math_classes/vec3.h"
#include "math_classes/color.h"
#include "math_classes/ray.h"

#include "camera.h"
#include "hittable/hittable.h"

#include <curand_kernel.h>

#define FLT_MAX 3.402823466e+38F // Define FLT_MAX if not already defined

__device__ inline vec3 get_sky_box_color(const ray& r) {
    // Simple skybox color based on ray direction
    // float t = 0.5f * (r.direction.y + 1.0f);
    // return (1.0f - t) * new_color(1.0f, 1.0f, 1.0f) + t * new_color(0.5f, 0.7f, 1.0f);
    return new_color(0.0f, 0.0f, 0.0f); // Default skybox color
}

__global__ void kernel(uint8_t* framebuffer, camera * cam, hittable * hittables, int hittable_count, curandState *states) {
    // Kernel code would go here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx > cam->cam_width || idy > cam->cam_height) return; // Prevent out-of-bounds access
    curandState *localState = &states[idy * cam->cam_width + idx]; // Get the random state for this pixel

    color pixel_color = new_color(0.0f, 0.0f, 0.0f); // Initialize pixel color to black

    color pixel_color_run;
    color sub_run_color;
    ray r;
    ray scattered;
    for(int s = 0; s < cam->samples_per_pixel; s++) {
        float x_offset = curand_uniform(localState);
        float y_offset = curand_uniform(localState);
        r = camera_get_ray(*cam, (float(idx)+x_offset) / (cam->cam_width), (float(cam->cam_height - idy - 1)+y_offset) / (cam->cam_height));


        hit_record hit_rec;
        hit_rec.t = FLT_MAX; // Initialize hit record
        int bnc = 0;
        bool hit_anything;
        pixel_color_run = new_color(1.0f, 1.0f, 1.0f); // full ray color
        for(; bnc < cam->max_bounces; bnc++) {
            hit_anything = false;

            
            for(int i = 0; i < hittable_count; i++) {
                if(hit(hittables[i], r, 0.0f, hit_rec.t, hit_rec, scattered, localState)) {
                    sub_run_color = hit_rec.ret_color; // If the ray hits a hittable object, use its color
                    hit_anything = true; // Mark that we hit something
                }
            }
            if (!hit_anything) {
                pixel_color_run = pixel_color_run * get_sky_box_color(r); // If no hit, use skybox color
                break;
            }
            else{
                if(!scatter(hit_rec.mat, r, hit_rec, sub_run_color, scattered, localState)) {
                    pixel_color_run = pixel_color_run * sub_run_color; 
                    break; // If scattering fails, stop the ray (emissives or absorbing materials)
                } 
                pixel_color_run = pixel_color_run * sub_run_color; // If hit, multiply by the color of the hittable object
                
                r = scattered; // Update ray direction to scattered direction
                inch_ray(r, 1e-6f); // Move the ray origin slightly forward to avoid self-intersection
                hit_rec.t = FLT_MAX; // Reset hit record for the next bounce
            }   
        }
        if(bnc == hittable_count){
            pixel_color_run = new_color(0.0f, 0.0f, 0.0f); // If no hittable objects were hit, set pixel color to black
        }

        pixel_color = pixel_color + pixel_color_run; // Accumulate color for this sample        
    }

    pixel_color = pixel_color / cam->samples_per_pixel; // Average the color over the number of samples
    

    // Convert color to uint8_t (assuming color is in range [0, 1])
    color_uint8 c_pixel_color = color_to_uint8(pixel_color);   
    // Copy 3 float values to the framebuffer
    framebuffer[(idy * cam->cam_width + idx) * 3 + 0] = c_pixel_color.r;
    framebuffer[(idy * cam->cam_width + idx) * 3 + 1] = c_pixel_color.g;
    framebuffer[(idy * cam->cam_width + idx) * 3 + 2] = c_pixel_color.b;
}

#endif // KERNEL_CODE_H