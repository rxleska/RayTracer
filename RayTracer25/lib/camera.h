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
};

__host__ __device__ inline ray camera_get_ray(const camera &self, float u, float v) {
    // Calculate the direction of the ray based on the viewport and focal length
    vec3 direction = {(u-0.5f) * self.viewport_width, (v-0.5f) * self.viewport_height, self.focal_length};
    return {self.origin, direction};
}

// class camera{
//     public: 
//         int cam_width;
//         int cam_height;
//         vec3 origin; // camera position
//         float focal_length; // focal length of the camera
//         float viewport_height; // height of the viewport
//         float viewport_width; // width of the viewport

//         __host__ __device__ inline camera(int width, int height, const vec3& o, float focal_len) 
//             : cam_width(width), cam_height(height), origin(o), focal_length(focal_len) {
//             viewport_height = 2.0f; // default viewport height
//             viewport_width = (float)cam_width / (float)cam_height * viewport_height; // calculate viewport width based on aspect ratio
//         }

//         __host__ __device__ inline ray get_ray(float u, float v) const {
//             // Calculate the direction of the ray based on the viewport and focal length
//             vec3 direction = vec3((u-0.5) * viewport_width, (v-0.5) * viewport_height, focal_length);
//             return ray(origin, direction);
//         }
// };

#endif // CAMERA_H