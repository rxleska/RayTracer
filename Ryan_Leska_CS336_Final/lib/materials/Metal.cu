#include "headers/Metal.hpp"


__device__ int Metal::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    Vec3 reflected = reflect(ray_in.direction, rec.normal);
    Vec3 fuzzMod = Vec3::random(-10,10,rand_state);
    fuzzMod.make_unit();
    fuzzMod =  fuzzMod * fuzz;
    scattered_out = Ray(rec.p, reflected + fuzzMod);
    attenuation = albedo;
    return 1;
}