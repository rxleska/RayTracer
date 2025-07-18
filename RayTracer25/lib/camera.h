#ifndef CAMERA_H
#define CAMERA_H

#include "math_classes/vec3.h"
#include "math_classes/ray.h"

// I want the camera to be definable on the host then copied to the device (but if that is not possible I will just define it on the device)

struct camera {
    int cam_width;
    int cam_height;
    vec3 origin; // camera position
    float focal_length; // focal length of the camera
    float viewport_height; // height of the viewport
    float viewport_width; // width of the viewport
    int samples_per_pixel; // number of samples per pixel for anti-aliasing
    int max_bounces; // maximum number of bounces for ray tracing 
    vec3 look_at_pos;
    vec3 w;
    vec3 u;
    vec3 v;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    __host__ __device__ inline ray get_ray(float u, float v) {
    // Calculate the direction of the ray based on the viewport and focal length
    // vec3 direction = {(u-0.5f) * self.viewport_width, (v-0.5f) * self.viewport_height, self.focal_length};
    // // transform the direction vector to account for the camera's angle vector
    // direction = self.u * direction.x + self.v * direction.y + self.w * direction.z; // Apply camera orientation
    vec3 direction = lower_left_corner + u * horizontal + v * vertical - origin; // Calculate the ray direction

    return {origin, direction};
}


    __host__ void calc_look_at(){
        w = unit_vector(origin - look_at_pos); // Calculate the w vector (camera direction)
        u = unit_vector(cross({0, 1, 0}, w)); // Calculate the u vector (camera right)
        v = cross(w, u); // Calculate the v vector (camera up)
        lower_left_corner = origin - u * (focal_length * viewport_width / 2.0f) - v * (focal_length * viewport_height / 2.0f) - w * focal_length; 
        horizontal = u * viewport_width * focal_length;
        vertical = v * viewport_height * focal_length;
    }
};




#endif // CAMERA_H