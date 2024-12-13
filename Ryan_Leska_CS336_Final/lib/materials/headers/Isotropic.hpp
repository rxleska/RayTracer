#ifndef ISOTROPIC_HPP
#define ISOTROPIC_HPP


#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

class Isotropic : public Material {
    public: 
        Vec3 albedo;
        __device__ Isotropic(Vec3 albedo) : albedo(albedo) {type = ISOTROPIC;}

        __device__ virtual int scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;
    
};

#endif