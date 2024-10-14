#ifndef PHONG_HPP
#define PHONG_HPP

#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

class Phong : public Material
{
    public:
        __device__ Phong(const Vec3 &albedo) : albedo(albedo) {type = PHONG;}
        __device__ Phong(const Vec3 &albedo, const Vec3 & kConsts, float a) : albedo(albedo), kConsts(kConsts), a(a) {type = PHONG;}

        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override; //filler function used to return that a phong material was hit
        __device__ Vec3 getColor(const Ray ray_out, const Vec3 *pointLights, int point_light_count, Vec3 ambient, curandState * rand_state) const;

    private:
        Vec3 albedo;
        Vec3 kConsts; //x -> ks, y -> kd, z -> ka
        float a;
};


#endif // PHONG_HPP