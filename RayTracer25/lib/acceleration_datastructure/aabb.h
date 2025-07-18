#ifndef AABB_H  
#define AABB_H

#include "../math_classes/ray.h"

// AABB (Axis-Aligned Bounding Box(es))

struct interval{
    float min;
    float max;
};

__host__ __device__ inline interval new_interval(float min, float max){
    return {min, max};
}

struct aabb {
    interval x;
    interval y;
    interval z;

    __host__ __device__ inline interval get_axis(int n){
        if(n==1) return y;
        if(n==2) return z;
        return x;
    }

    __host__ __device__ bool hit(const ray&r, interval ray_t) const {
        float t0, t1, adinv;

        // x axis
        adinv = 1.0f/r.direction.x;
        t0 = (x.min - r.origin.x) * adinv;
        t1 = (x.max - r.origin.x) * adinv;

        if(t0 < t1){
            if(t0 > ray_t.min) ray_t.min = t0;
            if(t1 < ray_t.min) ray_t.max = t1;
        } else {
            if(t1 > ray_t.min) ray_t.min = t1;
            if(t0 < ray_t.min) ray_t.max = t0;
        }
        if(ray_t.max <= ray_t.min) return false;

        // y axis
        adinv = 1.0f/r.direction.y;
        t0 = (y.min - r.origin.y) * adinv;
        t1 = (y.max - r.origin.y) * adinv;

        if(t0 < t1){
            if(t0 > ray_t.min) ray_t.min = t0;
            if(t1 < ray_t.min) ray_t.max = t1;
        } else {
            if(t1 > ray_t.min) ray_t.min = t1;
            if(t0 < ray_t.min) ray_t.max = t0;
        }
        if(ray_t.max <= ray_t.min) return false;

        // z axis
        adinv = 1.0f/r.direction.z;
        t0 = (z.min - r.origin.z) * adinv;
        t1 = (z.max - r.origin.z) * adinv;

        if(t0 < t1){
            if(t0 > ray_t.min) ray_t.min = t0;
            if(t1 < ray_t.min) ray_t.max = t1;
        } else {
            if(t1 > ray_t.min) ray_t.min = t1;
            if(t0 < ray_t.min) ray_t.max = t0;
        }
        if(ray_t.max <= ray_t.min) return false;

        return true;

    }

};

__host__ __device__ inline aabb new_aabb(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max){
    return {{x_min, x_max}, {y_min, y_max}, {z_min, z_max}};
}

#endif // AABB_H