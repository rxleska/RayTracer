#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.h"

#include "../math_classes/vec3.h"
#include "../math_classes/ray.h"
#include "../math_classes/color.h"

#include "../materials/material.h"

#include "../acceleration_datastructure/aabb.h"

struct sphere{
    vec3 center; // Center of the sphere
    float radius; // Radius of the sphere
    material mat;

    __device__ inline bool hit(ray& r, float t_min, float t_max, hit_record & record, ray& scattered, curandState* curandState) const {
    vec3 oc = center - r.origin;
    float a = dot(r.direction, r.direction);
    float b = -2.0 * dot(r.direction, oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 1e-6) {
        return false; // No intersection
    }
    float sqrt_discriminant = sqrtf(discriminant);
    // Find the nearest root that is within the bounds
    float root = (-b - sqrt_discriminant) / (2.0 * a);
    if (root < t_min || root > t_max) {
        root = (-b + sqrt_discriminant) / (2.0 * a);
        if (root < t_min || root > t_max) {
            return false; // No valid intersection
        }
    }
    record.t = root; // Set the hit distance
    record.intersection_point = r.origin + r.direction * root; // Calculate intersection point
    vec3 outward_normal = (record.intersection_point - center) / radius;
    record.is_front_face = dot(r.direction, outward_normal) < 0.0f;
    record.normal = record.is_front_face ? outward_normal : -outward_normal;
    record.mat = mat; 

    return true;
}
};

__host__ __device__ inline sphere new_sphere(const vec3& c, float r, const material& mat) {
    return {c, r, mat};
}




__host__ __device__ aabb get_bounding_box(const sphere& self){
    return {
        {self.center.x - self.radius, self.center.x + self.radius},
        {self.center.y - self.radius, self.center.y + self.radius},
        {self.center.z - self.radius, self.center.z + self.radius}
    };
}


#endif // SPHERE_H