#ifndef HITABLE_HPP
#define HITABLE_HPP

#include "../../processing/headers/Ray.hpp"
#include "HitRecord.hpp"

// class to represent a hitable object in the scene (abstract class, will never be instantiated)
class Hitable{
    public:
        // function to check if a ray hits the object
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
        __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const = 0;
        __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const = 0;
};

#endif