#ifndef LIGHT_HPP
#define LIGHT_HPP

#include "Material.hpp"

class Light : public Material {
    public:
        __device__ Light(const Vec3& color, float intensity) : color(color), intensity(intensity) { type = LIGHT; }

        __device__ virtual int scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;
        __device__ Vec3 emitted() const;

    
    private:
        Vec3 color;
        float intensity;
};


#endif // LIGHT_HPP