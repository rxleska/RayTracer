#ifndef HITTABLE_H
#define HITTABLE_H

#include "hit_record.h"

#include "sphere.h"
#include "polygon.h"

#include "../acceleration_datastructure/aabb.h"

enum hittable_type {
    SPHERE,
    POLYGON
};


struct hittable {
    hittable_type type; // Type of the hittable object
    union {
        sphere sphere_obj; // Sphere object
        polygon polygon_obj; // Polygon object
        // Add other hittable types here, e.g., polygon
    };

    __device__ inline bool hit(ray& r, float t_min, float t_max, hit_record& record, ray& scattered, curandState* curandState) const {
    switch (type) {
        case SPHERE:
            return sphere_obj.hit(r, t_min, t_max, record, scattered, curandState);
        case POLYGON:
            return polygon_obj.hit(r, t_min, t_max, record, scattered, curandState);
        // Add cases for other hittable types here
        default:
            return false; // Unsupported hittable type
    }
}
};



__device__ inline aabb get_bounding_box(const hittable& h){
    switch (h.type)
    {
        case SPHERE:
            return get_bounding_box(h.sphere_obj);
        case POLYGON:
            return get_bounding_box(h.polygon_obj);
        default:
            return {{0,0},{0,0},{0,0}};
    }
}


__host__ __device__ inline hittable new_hittable_sphere(const vec3& center, float radius, const material& mat) {
    hittable h;
    h.type = SPHERE;
    h.sphere_obj = new_sphere(center, radius, mat);
    return h;
}

__host__ __device__ inline hittable new_hittable_polygon(const vec3& v1,const vec3& v2,const vec3& v3, const material& mat) {
    hittable h;
    h.type = POLYGON;
    // calculate the normal of the polygon
    vec3 edge1 = v2 - v1;
    vec3 edge2 = v3 - v1;
    vec3 normal = unit_vector(cross(edge1, edge2)); // Calculate the normal vector of the polygon

    h.polygon_obj = {v1,v2,v3,normal,mat};
    return h;
}

#endif // HITTABLE_H