#include "headers/Ray.hpp"

__device__ Vec3 Ray::pointAt(float t) const {
    return origin + direction * t;
}

__device__ bool Ray::hitsBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float& t) const{
    // TODO see if there is a faster way to do this (I think I read about some optimizations for this)


    //find max t and min t for each x y z
    float tx1 = (x_min - origin.x) / direction.x;
    float tx2 = (x_max - origin.x) / direction.x;

    float max_t = fmax(tx1, tx2);
    float min_t = fmin(tx1, tx2);

    float ty1 = (y_min - origin.y) / direction.y;  
    float ty2 = (y_max - origin.y) / direction.y;

    max_t = fmin(fmax(ty1, ty2), max_t);
    min_t = fmax(fmin(ty1, ty2), min_t);

    float tz1 = (z_min - origin.z) / direction.z;
    float tz2 = (z_max - origin.z) / direction.z;

    max_t = fmin(fmax(tz1, tz2), max_t);
    min_t = fmax(fmin(tz1, tz2), min_t);

    t = min_t;
    return max_t >= min_t;
}