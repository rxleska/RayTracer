#include "headers/Dielectric.hpp"

//min function for floats
#ifndef d_fmin
#define d_fmin(a,b) ((a < b) ? a : b)
#endif

__device__ int Dielectric::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    attenuation = Vec3(1.0, 1.0, 1.0);
    float refraction_ratio = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    Vec3 unit_direction = ray_in.direction.normalized();

    float cos_theta = d_fmin((unit_direction * -1.0f).dot(rec.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3 direction;

    if (cannot_refract || schlick(cos_theta, refraction_ratio) > 0.5) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    scattered_out = Ray(rec.p, direction);
    return 1;  
}