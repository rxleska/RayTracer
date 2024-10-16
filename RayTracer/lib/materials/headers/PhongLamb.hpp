#ifndef PHONGLAMB_HPP
#define PHONGLAMB_HPP

#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

class PhongLamb : public Material
{
    public:
        __device__ PhongLamb(const Vec3 &albedo) : albedo(albedo), kConsts(Vec3(0.3,0.5,0.2)), a(0.1), bc(10) {type = PHONGLAMB;}
        __device__ PhongLamb(const Vec3 &albedo, const Vec3 & kConsts, float a) : albedo(albedo), kConsts(kConsts), a(a), bc(10) {type = PHONGLAMB;}
        __device__ PhongLamb(const Vec3 &albedo, const Vec3 & kConsts, float a, int bc) : albedo(albedo), kConsts(kConsts), a(a), bc(bc) {type = PHONGLAMB;}

        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override; //filler function used to return that a phong material was hit

        Vec3 kConsts; //x -> ks, y -> kd, z -> ka
        float a;
        Vec3 albedo;
        int bc;
        
};


#endif // PHONGLAMB_HPP