#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "../math_classes/color.h"
#include "../math_classes/ray.h"
#include "../math_classes/vec3.h"
#include "../hittable/hit_record.h"

struct lambertian {
    color albedo; // Diffuse color of the material
};

__device__ inline bool scatter(const lambertian& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    vec3 scatter_direction = rec.normal + random_unit_vector(curandState); // Scatter in a random direction around the normal
    // If the scatter direction is near zero, use the normal as the scatter direction
    if (is_near_zero(scatter_direction)) {
        scatter_direction = rec.normal;
    }
    scatter_direction = unit_vector(scatter_direction); // Normalize the scatter direction
    
    scattered = {(rec.intersection_point + rec.normal * 1e-4f), scatter_direction}; // Create a new ray from the intersection point in the scatter direction
    attenuation = self.albedo; // Set the attenuation color to the material's albedo
    return true; // Return true to indicate that scattering occurred
}

#endif // LAMBERTIAN_H