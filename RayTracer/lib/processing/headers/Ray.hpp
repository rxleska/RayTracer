#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

class Ray{
    public:
        Vec3 origin;
        Vec3 direction;

        Ray() : origin(Vec3()), direction(Vec3()) {}
        Ray(Vec3 origin, Vec3 direction) : origin(origin), direction(direction.normalized()) {}

        Vec3 pointAt(float t) const;
};

#endif