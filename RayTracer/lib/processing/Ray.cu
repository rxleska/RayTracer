#include "headers/Ray.hpp"

__device__ Vec3 Ray::pointAt(float t) const {
    return origin + direction * t;
}

__device__ bool Ray::hitsBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float& t) const {
    float tmin,tmax,tymin,tymax,tzmin,tzmax;    

    if(direction.x >= 0){
        tmin = (x_min - origin.x) / direction.x;
        tmax = (x_max - origin.x) / direction.x;
    }
    else{
        tmin = (x_max - origin.x) / direction.x;
        tmax = (x_min - origin.x) / direction.x;
    }

    if(direction.y >= 0){
        tymin = (y_min - origin.y) / direction.y;
        tymax = (y_max - origin.y) / direction.y;
    }
    else{
        tymin = (y_max - origin.y) / direction.y;
        tymax = (y_min - origin.y) / direction.y;
    }

    if((tmin > tymax) || (tymin > tmax)){
        return false;
    }

    if(tymin > tmin){
        tmin = tymin;
    }

    if(tymax < tmax){
        tmax = tymax;
    }

    if(direction.z >= 0){
        tzmin = (z_min - origin.z) / direction.z;
        tzmax = (z_max - origin.z) / direction.z;
    }
    else{
        tzmin = (z_max - origin.z) / direction.z;
        tzmax = (z_min - origin.z) / direction.z;
    }

    if((tmin > tzmax) || (tzmin > tmax)){
        return false;
    }

    if(tzmin > tmin){
        tmin = tzmin;
    }

    if(tzmax < tmax){
        tmax = tzmax;
    }

    t = tmax;
    return true;
}

__device__ Vec3 Ray::inverse() const{
    return Vec3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
}
