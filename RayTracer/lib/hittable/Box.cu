#include "headers/Box.hpp"

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

#include "../materials/headers/Textured.hpp"

#include "../materials/headers/Lambertian.hpp"

__device__ float Box::bound(int axis, int side) const{
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

__device__ bool Box::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    
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

    if (entry > exit || exit < t_min || entry > t_max) {
        return false;
    }

    rec.t = entry;
    rec.p = r.origin + r.direction * entry;

    Vec3 normal;
    if (entry == tmin.x) {
        if(r.direction.x > 0){
            normal = Vec3(-1,0,0);
        }
        else{
            normal = Vec3(1,0,0);
        }
        
    } else if (entry == tmin.y) {
        if(r.direction.y > 0){
            normal = Vec3(0,-1,0);
        }
        else{
            normal = Vec3(0,1,0);
        }
    } else {
        if(r.direction.z > 0){
            normal = Vec3(0,0,-1);
        }
        else{
            normal = Vec3(0,0,1);
        }
    }

    rec.front_face = false;
    rec.normal = normal;
    rec.mat = this->mat;

    return true;
}

__device__ void Box::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const{
    x_min = min.x;
    x_max = max.x;
    y_min = min.y;
    y_max = max.y;
    z_min = min.z;
    z_max = max.z;
}

__device__ bool Box::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const{
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

__device__ Vec3 Box::getRandomPointInHitable(curandState *state) const {
    return Vec3(
        curand_uniform(state) * (max.x - min.x) + min.x,
        curand_uniform(state) * (max.y - min.y) + min.y,
        curand_uniform(state) * (max.z - min.z) + min.z
    );
}

__device__ float Box::get2dArea() const {
    Vec3 diff = max - min;
    return diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
}

#include <iostream>

__device__ void Box::debug_print() const{
    printf("Box: min: (%f, %f, %f) max: (%f, %f, %f)\n", min.x, min.y, min.z, max.x, max.y, max.z);
}