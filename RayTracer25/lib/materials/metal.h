#ifndef METAL_H
#define METAL_H

#include "../math_classes/color.h"
#include "../math_classes/ray.h"    
#include "../math_classes/vec3.h"
#include "../hittable/hit_record.h"
#include "material.h"

struct metal {
    color albedo; // Reflective color of the metal
    float fuzz; // Fuzziness factor for the metal surface
};

__device__ inline bool scatter(const metal& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal); // Reflect the incoming ray direction around the normal
    vec3 fuzzMod = random_unit_vector(curandState) * self.fuzz; // Add fuzziness to the reflection

    scattered = {rec.intersection_point + rec.normal * 1e-4f, reflected + fuzzMod}; 
    attenuation = self.albedo; // Set the attenuation color to the material's albedo

    return true; // Return true to indicate that scattering occurred
}


#endif // METAL_H