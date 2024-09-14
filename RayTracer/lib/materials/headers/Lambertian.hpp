#ifndef LAMBERTIAN_HPP
#define LAMBERTIAN_HPP

#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

class Lambertian : public Material
{
    public:
        __device__ Lambertian(const Vec3 &albedo) : albedo(albedo) {type = LAMBERTIAN;}

        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private:
        Vec3 albedo;
};


#endif