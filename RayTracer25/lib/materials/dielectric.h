#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "../math_classes/vec3.h"
#include "../math_classes/ray.h"
#include "../hittable/hit_record.h"
#include "material.h"

struct dielectric {
    float ref_idx; // Refractive index of the dielectric material
};



__device__ inline bool scatter(const dielectric& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    attenuation = new_color(1.0f, 1.0f, 1.0f); // Attenuation is always white for dielectric materials
    float refraction_ratio = rec.is_front_face ? (1.0f / self.ref_idx) : self.ref_idx; // Determine the refraction ratio based on the front face of the hit record

    vec3 unit_direction = unit_vector(r_in.direction); // Normalize the incoming ray direction
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f); // Calculate the cosine of the angle between the ray direction and the normal
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f; // Check if total internal reflection occurs
    vec3 direction;
    // if (cannot_refract || schlick(cos_theta, refraction_ratio) > curand_uniform(curandState)) {
    //     // Reflect the ray if total internal reflection occurs or based on Schlick's approximation
    //     direction = reflect(unit_direction, rec.normal);
    // } else {
        // Refract the ray if it can pass through the material
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    // }

    scattered = {rec.intersection_point, direction}; // Create a new ray from the intersection point in the calculated direction
    return true; // Return true to indicate that scattering occurred
}

#endif // DIELECTRIC_H