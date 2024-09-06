#include "headers/material.hpp"


__device__ int material::scatter( // 0:no scatter, 1:scatter, 2:scatter and absorb
    const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const
{
    return false;
}
__device__ bool material::getIsLightSrc() const { return isLightSrc; }
__device__ void material::setIsLightSrc(bool isLightSrc) { this->isLightSrc = isLightSrc; }

// defining some materials

// Lambertian material - scatters light in all directions
__device__ int lambertian::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    auto scatter_direction = rec.normal + random_unit_vector();

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return 1;
}

// Metal material - scatters light in a single direction with reflection
__device__ int metal::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0 ? 1 : 0);
}

// Dielectric material - scatters light in a single direction with refraction
__device__ int dielectric::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    attenuation = color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    //TODO 16 will be wrong if the # thread vertically changes
    //get curand state
    // curandState local_rand_state = d_rand_state[threadIdx.x + threadIdx.y * 16];

    if (cannot_refract || reflectance(cos_theta, ri) >  0.5)
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, direction);
    return 1;
}


// Light material - scatters light in all directions (inverse: absorbs inverse rays)
__device__ int light::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    return 2;
}

__device__ color light::emitted() const { return emission; }