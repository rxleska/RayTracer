#ifndef HITABLE_HPP
#define HITABLE_HPP

#include "../../processing/headers/Ray.hpp"
#include "HitRecord.hpp"

// class to represent a hitable object in the scene (abstract class, will never be instantiated)
class Hitable{
    public:
        // function to check if a ray hits the object
        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

#endif