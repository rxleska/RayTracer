#ifndef SCENE_HPP
#define SCENE_HPP

#include "HitRecord.hpp"
#include "Hitable.hpp"

class Scene{
    public:
        __device__ Scene();
        __device__ Scene(Hitable **hitables, int hitable_count);
        __host__ void free_memory();
        __device__ void resize(int new_capacity);
        __device__ void addHitable(Hitable *hittable);
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
        
        
        int hitable_count;

        Hitable **hitables;
        int hitable_capacity;
};


#endif