#include "headers/Sphere.hpp"
#include <cmath>


#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../materials/headers/Textured.hpp"


__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
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

    //if normal and ray direction are close to 90 degrees, set edge_hit to true
    rec.edge_hit = rec.normal.dot(r.direction) < 0.01f;

    rec.mat = mat; 

    if(mat->type == MaterialType::TEXTURED){
        rec.u = ((Textured*)mat)->rot + atan2(rec.normal.z, rec.normal.x) / (2.0f * M_PI);
        rec.v = 0.5f - asin(rec.normal.y) / M_PI;
        rec.edge_hit = true; // increase anti-aliasing always on textured spheres
    }

    return true;
}

__device__ void Sphere::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const{
    x_min = center.x - radius;
    x_max = center.x + radius;
    y_min = center.y - radius;
    y_max = center.y + radius;
    z_min = center.z - radius;
    z_max = center.z + radius;
}


__device__ bool Sphere::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const{
    return (
        // check if sphere mins are below box maxes
        // check if sphere maxes are above box mins
        (center.x - radius <= x_max + F_EPSILON) &&
        (center.x + radius >= x_min - F_EPSILON) &&
        (center.y - radius <= y_max + F_EPSILON) &&
        (center.y + radius >= y_min - F_EPSILON) &&
        (center.z - radius <= z_max + F_EPSILON) &&
        (center.z + radius >= z_min - F_EPSILON) 
    );
}

__device__ void Sphere::debug_print() const{
    printf("Sphere: center: (%f, %f, %f) radius: %f\n", center.x, center.y, center.z, radius);
}


__device__ Vec3 Sphere::getRandomPointInHitable(curandState *state) const {
    float theta = 2 * M_PI * curand_uniform(state);
    float phi = acos(1 - 2 * curand_uniform(state));
    float x = center.x + radius * sin(phi) * cos(theta);
    float y = center.y + radius * sin(phi) * sin(theta);
    float z = center.z + radius * cos(phi);
    return Vec3(x, y, z);    
}

__device__ float Sphere::get2dArea() const{
    return 2.0f * M_PI * radius * radius;
}