#ifndef TEXTURED_HPP
#define TEXTURED_HPP

#include "Material.hpp"

// textured material class 
class Textured : public Material{
    public:
        __device__ Textured(Vec3 *texture, int width, int height) : texture(texture), width(width), height(height) {type = TEXTURED;}
        __device__ virtual int scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const override;

    private: 
        Vec3 *texture; //array of length width * height
        int width;
        int height;
};


__host__ float *load_texture(const char *filename, int &width, int &height);


#endif