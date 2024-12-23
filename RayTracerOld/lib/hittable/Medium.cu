#include "headers/Medium.hpp"

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

#include "../hittable/headers/HitRecord.hpp"

#include "../materials/headers/Textured.hpp"

#include "../materials/headers/Lambertian.hpp"

__device__ float Medium::bound(int axis, int side) const{
    if(axis == 0){
        if(side == 0){
            return min.x;
        }
        else{
            return max.x;
        }
    }
    else if(axis == 1){
        if(side == 0){
            return min.y;
        }
        else{
            return max.y;
        }
    }
    else{
        if(side == 0){
            return min.z;
        }
        else{
            return max.z;
        }
    }
}

#include <cassert>

__device__ bool Medium::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
    Vec3 inverse = r.inverse();
    Vec3 tmin = (min - r.origin) * inverse;
    Vec3 tmax = (max - r.origin) * inverse;
    if(tmin.x > tmax.x){
        float temp = tmin.x;
        tmin.x = tmax.x;
        tmax.x = temp;
    }
    if(tmin.y > tmax.y){
        float temp = tmin.y;
        tmin.y = tmax.y;
        tmax.y = temp;
    }
    if(tmin.z > tmax.z){
        float temp = tmin.z;
        tmin.z = tmax.z;
        tmax.z = temp;
    }
    
    float entry = fmax(fmax(tmin.x, tmin.y), tmin.z);
    float exit = fmin(fmin(tmax.x, tmax.y), tmax.z);

    if (entry < t_min){
        entry = t_min;
    }
    if (exit > t_max){
        exit = t_max;
    }
    if(entry >= exit){
        return false;
    }
    if(entry < 0){
        entry = 0;
    }

    float hit_dist = neg_inv_density * logf(curand_uniform(state));
    if(hit_dist > (exit - entry)){
        return false;
    }

    rec.t = entry + hit_dist;
    rec.p = r.pointAt(rec.t);
    rec.normal = Vec3(1, 0, 0); // arbitrary normal
    rec.front_face = true;
    rec.mat = mat;

    return true;
}

__device__ void Medium::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const{
    x_min = min.x;
    x_max = max.x;
    y_min = min.y;
    y_max = max.y;
    z_min = min.z;
    z_max = max.z;
}

__device__ bool Medium::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const{
    return (
        // check if box mins are below box maxes
        // check if box maxes are above box mins
        (min.x <= x_max + F_EPSILON) &&
        (max.x >= x_min - F_EPSILON) &&
        (min.y <= y_max + F_EPSILON) &&
        (max.y >= y_min - F_EPSILON) &&
        (min.z <= z_max + F_EPSILON) &&
        (max.z >= z_min - F_EPSILON) 
    );
}

__device__ Vec3 Medium::getRandomPointInHitable(curandState *state) const {
    return Vec3(
        curand_uniform(state) * (max.x - min.x) + min.x,
        curand_uniform(state) * (max.y - min.y) + min.y,
        curand_uniform(state) * (max.z - min.z) + min.z
    );
}

__device__ float Medium::get2dArea() const {
    Vec3 diff = max - min;
    return diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
}

#include <iostream>

__device__ void Medium::debug_print() const{
    printf("Box: min: (%f, %f, %f) max: (%f, %f, %f)\n", min.x, min.y, min.z, max.x, max.y, max.z);
}