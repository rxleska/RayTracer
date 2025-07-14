#ifndef MATERIAL_H
#define MATERIAL_H

#include "lambertian.h"

enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

struct material {
    material_type type; // Type of the material
    union {
        lambertian lambertian_obj; // Lambertian material
        // Add other material types here, e.g., metal, dielectric
    };
};

__device__ bool scatter(const material& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    switch (self.type) {
        case LAMBERTIAN:
            return scatter(self.lambertian_obj, r_in, rec, attenuation, scattered, curandState);
        // Add cases for other material types here
        default:
            return false; // Unsupported material type
    }
}

// Function to create a new material of type lambertian
__host__ __device__ inline material new_material_lambertian(const color& albedo) {
    material m;
    m.type = LAMBERTIAN;
    m.lambertian_obj = {albedo}; // Initialize the lambertian object with the given albedo
    return m;
}

#endif // MATERIAL_H