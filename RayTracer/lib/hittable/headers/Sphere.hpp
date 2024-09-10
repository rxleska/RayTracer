#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Hitable.hpp"

// class to represent a sphere object in the scene (inherits from Hitable)
class Sphere: public Hitable{
    public:
        Vec3 center;
        float radius;


        // blank constructor
        Sphere() : center(Vec3()), radius(0) {}

        // constructor with parameters
        Sphere(Vec3 center, float radius) : center(center), radius(radius) {}

        // function to check if a ray hits the sphere
        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
};



#endif