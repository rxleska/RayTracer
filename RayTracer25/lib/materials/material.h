#ifndef MATERIAL_H
#define MATERIAL_H

#include "lambertian.h"
#include "metal.h"
#include "dielectric.h"

enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

struct material {
    material_type type; // Type of the material
    union {
        lambertian lambertian_obj; // Lambertian material
        metal metal_obj; // Metal material
        dielectric dielectric_obj; // Dielectric material
        // Add other material types here, e.g., metal, dielectric
    };
};

__device__ inline bool scatter(const material& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    switch (self.type) {
        case LAMBERTIAN:
            return scatter(self.lambertian_obj, r_in, rec, attenuation, scattered, curandState);
        case METAL:
            return scatter(self.metal_obj, r_in, rec, attenuation, scattered, curandState);
        case DIELECTRIC:
            return scatter(self.dielectric_obj, r_in, rec, attenuation, scattered, curandState);
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

__host__ __device__ inline material new_material_metal(const color& albedo, float fuzz) {
    material m;
    m.type = METAL;
    m.metal_obj = {albedo, fuzz}; // Initialize the metal object with the given albedo and fuzziness
    return m;
}

__host__ __device__ inline material new_material_dielectric(float ref_idx) {
    material m;
    m.type = DIELECTRIC;
    m.dielectric_obj = {ref_idx}; // Initialize the dielectric object with the given refractive index
    return m;
}

#endif // MATERIAL_H