#include "headers/Sphere.hpp"
#include <cmath>

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    // using the quadratic formula to solve for t in the equation of a ray-sphere intersection

    //formula for sphere ray intersection taken from wikipedia https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    float b = (r.direction.dot((r.origin - center)) * 2.0f);
    float discriminant = b*b - ((r.direction.mag2() * ((r.origin - center).mag2() - radius*radius)) * 4.0f); //4ac

    if(discriminant < 0.000001) return false;

    float t = (-b - std::sqrt(discriminant)) / (2.0f * r.direction.mag2());
    if(t < t_min || t > t_max){
        t = (-b + std::sqrt(discriminant)) / (2.0f * r.direction.mag2());
        if(t < t_min || t > t_max) return false;
    }

    rec.t = t;
    rec.p = r.pointAt(t);
    rec.normal = (rec.p - center) / radius;
    rec.front_face = rec.normal.dot(r.direction) < F_EPSILON;
    rec.normal = rec.front_face ? rec.normal : rec.normal * -1.0f;
    rec.mat = mat; 
    return true;
}
