#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "../math_classes/vec3.h"
#include "../math_classes/color.h"
#include "../math_classes/ray.h"

#include "../materials/material.h"

struct material; // Forward declaration of material struct


struct hit_record {
    vec3 intersection_point; // Point of intersection
    vec3 normal; // Normal at the intersection point
    float t; // Distance along the ray to the intersection point
    color ret_color; // Color of the surface at the intersection point
    bool is_front_face; // Whether the hit is on the front face of the surface
    material mat; // Material of the surface at the intersection point
};

__host__ __device__ inline hit_record new_hit_record(const vec3& point, const vec3& normal, float t) {
    return {point, normal, t, true, NULL};
}


#endif // HIT_RECORD_H