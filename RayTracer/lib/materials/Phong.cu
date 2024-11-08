#include "headers/Phong.hpp"

__device__ int Phong::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    // phong is handled in a phong function in scene, since it needs multiple parts of the scene to calculate the color
    return 3;
}
