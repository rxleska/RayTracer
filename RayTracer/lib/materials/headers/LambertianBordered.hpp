#ifndef LAMBERTIAN_BORDERED_HPP
#define LAMBERTIAN_BORDERED_HPP

#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

class LambertianBordered : public Material
{
    public:
        __device__ LambertianBordered(const Vec3 &albedo) : albedo(albedo) {type = LAMBERTIAN;}
        __device__ LambertianBordered(const Vec3 &albedo, const Vec3 &border) : albedo(albedo), border_color(border) {type = LAMBERTIAN;}

        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private:
        Vec3 albedo;
        Vec3 border_color = Vec3(0.0f, 0.0f, 0.0f);
};


#endif