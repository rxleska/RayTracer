#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "../../processing/headers/Ray.hpp"
#include "../../hittable/headers/HitRecord.hpp"

// forward declaration of HitRecord class
class HitRecord;

// Material Types Enum
enum MaterialType
{
    LAMBERTIAN, //everywhere scatter
    METAL, // reflect
    DIELECTRIC, // refract
    LIGHT // emit light
};


// abstract Material class
class Material
{
    public:
        MaterialType type;
        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const = 0;

        
};




#endif