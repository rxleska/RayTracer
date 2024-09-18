#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Hitable.hpp"

// class to represent a sphere object in the scene (inherits from Hitable)
class Sphere: public Hitable{
    public:
        Vec3 center;
        float radius;
        Material * mat;
        float area; //stored for faster computation


        // blank constructor
        __device__ Sphere() : center(Vec3()), radius(0) {}

        // constructor with parameters
        __device__ Sphere(Vec3 center, float radius, Material * mat) : center(center), radius(radius), mat(mat) {}

        // function to check if a ray hits the sphere
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

        // function to get the bounding box of the sphere
        __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const;
};



#endif