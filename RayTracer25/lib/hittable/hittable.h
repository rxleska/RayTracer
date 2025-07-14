#ifndef HITTABLE_H
#define HITTABLE_H

#include "hit_record.h"

#include "sphere.h"

enum hittable_type {
    SPHERE,
    POLYGON
};


struct hittable {
    hittable_type type; // Type of the hittable object
    union {
        sphere sphere_obj; // Sphere object
        // Add other hittable types here, e.g., polygon
    };
};

__device__ inline bool hit(const hittable& h, ray& r, float t_min, float t_max, hit_record& record, ray& scattered, curandState* curandState) {
    switch (h.type) {
        case SPHERE:
            return hit(h.sphere_obj, r, t_min, t_max, record, scattered, curandState);
        // Add cases for other hittable types here
        default:
            return false; // Unsupported hittable type
    }
}


__host__ __device__ inline hittable new_hittable_sphere(const vec3& center, float radius, const material& mat) {
    hittable h;
    h.type = SPHERE;
    h.sphere_obj = new_sphere(center, radius, mat);
    return h;
}

#endif // HITTABLE_H