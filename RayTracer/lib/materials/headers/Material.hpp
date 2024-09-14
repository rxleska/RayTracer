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

//min function for floats
#ifndef d_fmin
#define d_fmin(a,b) ((a < b) ? a : b)
#endif

//max function for floats
#ifndef d_fmax 
#define d_fmax(a,b) ((a > b) ? a : b)
#endif

// abstract Material class
class Material
{
    public:
        MaterialType type;
        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const = 0;

        __device__ inline Vec3 refract(const Vec3 &uv, const Vec3 &n, float ni_over_nt) const
        {
            float cos_theta = fminf((uv*-1.0f).dot(n), 1.0f);
            Vec3 r_out_perp = (uv + n * cos_theta) * ni_over_nt;
            Vec3 r_out_parallel = n * -sqrtf(fmax(0.0f, 1.0f - r_out_perp.mag2()));
            return r_out_perp + r_out_parallel;
        }
        
        //reflective function
        __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n) const
        {
            return v -  (n * (2 * v.dot(n)));
        }

        //reflectance function
        __device__ inline float schlick(float cosine, float ref_idx) const
        {
            float r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine), 5);
        }
};




#endif