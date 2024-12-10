#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

class Ray{
    public:
        Vec3 origin;
        Vec3 direction;

        __device__ Ray() : origin(Vec3()), direction(Vec3()) {}
        __device__ Ray(Vec3 origin, Vec3 direction) : origin(origin), direction(direction.normalized()) {}
        
        //copy constructor
        __device__ Ray(const Ray& r) : origin(r.origin), direction(r.direction) {}


        __device__ Vec3 pointAt(float t) const;
        __device__ bool hitsBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float& t) const;

        __device__ Vec3 inverse() const;

        __device__ void move3Epsilon();
};

#endif