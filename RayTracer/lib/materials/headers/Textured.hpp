#ifndef TEXTURED_HPP
#define TEXTURED_HPP

#include "Material.hpp"

// textured material class 
class Textured : public Material{
    public:
        __device__ Textured(float *texture, int width, int height) : texture(texture), width(width), height(height) {type = TEXTURED;}
        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private: 
        float *texture;
        int width;
        int height;
};


#endif