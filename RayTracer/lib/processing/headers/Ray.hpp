#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

class Ray{
    public:
        Vec3 origin;
        Vec3 direction;

        __device__ Ray() : origin(Vec3()), direction(Vec3()) {}
        __device__ Ray(Vec3 origin, Vec3 direction) : origin(origin), direction(direction.normalized()) {}

        __device__ Vec3 pointAt(float t) const;
};

#endif