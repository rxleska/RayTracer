#include "headers/Box.hpp"

#ifndef F_EPSILON
#define F_EPSILON 0.000001
#endif

#include "../materials/headers/Textured.hpp"

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
    float tmin,tmax,tymin,tymax,tzmin,tzmax;
    
    Vec3 inverse = r.inverse();
    bool sign[3] = {inverse.x < 0, inverse.y < 0, inverse.z < 0};

    tmin = (bound(0, sign[0]) - r.origin.x) * inverse.x;
    tmax = (bound(0, 1 - sign[0]) - r.origin.x) * inverse.x;
    tymin = (bound(1, sign[1]) - r.origin.y) * inverse.y;
    tymax = (bound(1, 1 - sign[1]) - r.origin.y) * inverse.y;
    if( 
        (tmin > tymax) ||
        (tymin > tmax)
    ){
        return false;
    }
    if(tymin > tmin){
        tmin = tymin;
        
    }
    if(tymax < tmax){
        tmax = tymax;
    }
    tzmin = (bound(2, sign[2]) - r.origin.z) * inverse.z;
    tzmax = (bound(2, 1 - sign[2]) - r.origin.z) * inverse.z;
    if( 
        (tmin > tzmax) ||
        (tzmin > tmax)
    ){
        return false;
    }
    if(tzmin > tmin){
        tmin = tzmin;
    }
    if(tzmax < tmax){
        tmax = tzmax;
    }
    
    // update the hit record
    rec.t = tmin;
    rec.p = r.pointAt(tmin);
    Vec3 normal = Vec3(0, 0, 0);

    if(fabs(rec.p.x - min.x) < F_EPSILON){
        normal = normal + Vec3(-1, 0, 0);
    }
    if(fabs(rec.p.x - max.x) < F_EPSILON){
        normal = normal + Vec3(1, 0, 0);
    }
    if(fabs(rec.p.y - min.y) < F_EPSILON){
        normal = normal + Vec3(0, -1, 0);
    }
    if(fabs(rec.p.y - max.y) < F_EPSILON){
        normal = normal + Vec3(0, 1, 0);
    }
    if(fabs(rec.p.z - min.z) < F_EPSILON){
        normal = normal + Vec3(0, 0, -1);
    }
    if(fabs(rec.p.z - max.z) < F_EPSILON){
        normal = normal + Vec3(0, 0, 1);
    }

    rec.normal = normal;



    rec.front_face = rec.normal.dot(r.direction) < F_EPSILON;//I dont think this can be zero
    // rec.normal = rec.front_face ? rec.normal : rec.normal * -1.0f;
    rec.edge_hit = false; //TODO edge hits on boxes
    rec.mat = mat;

    return (tmin < t_max - F_EPSILON) && (tmax > t_min + F_EPSILON);
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