#include "headers/ray.hpp"

__device__ ray::ray() {}
__device__ ray::ray(const vec3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}


__device__ vec3 ray::origin() const { return orig; }
__device__ vec3 ray::direction() const { return dir; }
__device__ vec3 ray::at(double t) const { return orig + t * dir; }
    