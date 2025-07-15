#ifndef RAY_H
#define RAY_H

#include "vec3.h"

// using point3 = vec3; // Define point3 as a vec3 for 3D points

struct ray{
    vec3 origin; // Ray origin
    vec3 direction; // Ray direction
};

__host__ __device__ inline ray new_ray(const vec3& o, const vec3& d) {
    return {o, d};
}

__host__ __device__ inline vec3 ray_at(const ray& r, float t) {
    return { 
        r.origin.x + r.direction.x * t, 
        r.origin.y + r.direction.y * t, 
        r.origin.z + r.direction.z * t 
    };
}

__device__ inline void inch_ray(ray&r, const float t) {
    r.origin.x += r.direction.x * t;
    r.origin.y += r.direction.y * t;
    r.origin.z += r.direction.z * t;
}

// class ray {
//     public:
//         vec3 origin; 
//         vec3 direction;

//         __host__ __device__ inline ray() : origin(vec3()), direction(vec3()) {}
//         __host__ __device__ inline ray(const vec3& o, const vec3& d) : origin(o), direction(d) {}

//         // Getters (not preferred on the device, I would rather use origin.e[0], direction.e[0] directly) // TODO Look into using __forceinline for these getters and setters (if I add them)
//         __host__ __device__ inline vec3 get_origin() const { return origin; }
//         __host__ __device__ inline vec3 get_direction() const { return direction; }

//         // Get point at parameter t along the ray
//         __host__ __device__ inline vec3 at(float t) const {
//             return { origin.x + direction.x * t, 
//                      origin.y + direction.y * t, 
//                      origin.z + direction.z * t};
//         }        
// };


#endif