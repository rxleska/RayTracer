#ifndef DIELECTRIC_HPP
#define DIELECTRIC_HPP


#include "Material.hpp"
#include "../../hittable/headers/HitRecord.hpp"
#include "../../processing/headers/Ray.hpp"

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

class Dielectric : public Material
{
    public:
        __device__ Dielectric(float refraction_index) : refraction_index(refraction_index) {type = DIELECTRIC;}

        __device__ virtual int scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private:
        float refraction_index;
};

#endif
        