#ifndef HIT_RECORD_HPP
#define HIT_RECORD_HPP

#include "../../processing/headers/Vec3.hpp"

// class to represent a hit record (when a ray hits an object)
class HitRecord{
    public:
        // time of hit (this is the time along a ray)
        float t;
        // point of hit
        Vec3 p;
        // normal of the hit object at the point of hit (used for reflection/refraction/absorption/etc)
        Vec3 normal;

        // blank constructor
        HitRecord() : t(0), p(Vec3()), normal(Vec3()) {}

        // constructor with parameters
        HitRecord(float t, Vec3 p, Vec3 normal) : t(t), p(p), normal(normal) {}
}; 


#endif