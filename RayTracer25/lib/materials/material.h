#ifndef MATERIAL_H
#define MATERIAL_H

struct lambertian {
    color albedo; // Diffuse color of the material
};

struct metal {
    color albedo; // Reflective color of the metal
    float fuzz; // Fuzziness factor for the metal surface
};

struct dielectric {
    float ref_idx; // Refractive index of the dielectric material
};

struct emissive {
    color emit_color; // Emissive color of the material
    float intensity; // Intensity of the emission
};

enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE
};

struct material {
    material_type type; // Type of the material
    union {
        lambertian lambertian_obj; // Lambertian material
        metal metal_obj; // Metal material
        dielectric dielectric_obj; // Dielectric material
        emissive emissive_obj; // Emissive material
        // Add other material types here, e.g., metal, dielectric
    };
};

#include "lambertian.h"
#include "metal.h"
#include "dielectric.h"


__device__ inline bool scatter(const material& self, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* curandState) {
    switch (self.type) {
        case LAMBERTIAN:
            return scatter(self.lambertian_obj, r_in, rec, attenuation, scattered, curandState);
        case METAL:
            return scatter(self.metal_obj, r_in, rec, attenuation, scattered, curandState);
        case DIELECTRIC:
            return scatter(self.dielectric_obj, r_in, rec, attenuation, scattered, curandState);
        case EMISSIVE:
            attenuation = self.emissive_obj.emit_color * self.emissive_obj.intensity; // Set attenuation to the emissive color
            scattered = {rec.intersection_point, new_vec3(0.0f, 0.0f, 0.0f)}; // No scattering for emissive materials
            return false; // Return true to indicate that scattering occurred
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

__host__ __device__ inline material new_material_emissive(const color& emit_color, float intensity) {
    material m;
    m.type = EMISSIVE;
    m.emissive_obj = {emit_color, intensity}; // Initialize the emissive object with the given color and intensity
    return m;
}

#endif // MATERIAL_H