#include "headers/Isotropic.hpp"
#include "../processing/headers/Vec3.hpp"

__device__ int Isotropic::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *rand_state) const {
    attenuation = albedo;
    scattered = Ray(rec.p, Vec3::random(-10, 10, rand_state).normalized());
    return 1;
}