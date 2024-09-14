#ifndef METAL_HPP
#define METAL_HPP

#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

class Metal : public Material
{
    public:
        __device__ Metal(const Vec3 &albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {type = METAL;}

        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private:
        Vec3 albedo;
        float fuzz; //noise on the surface
};

#endif