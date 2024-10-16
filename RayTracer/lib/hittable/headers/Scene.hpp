#ifndef SCENE_HPP
#define SCENE_HPP

#include "HitRecord.hpp"
#include "Hitable.hpp"
#include "../../processing/headers/Camera.hpp"

class Scene{
    public:
        __device__ Scene();
        __device__ Scene(Hitable **hitables, int hitable_count);
        __host__ void free_memory();
        __device__ void resize(int new_capacity);
        __device__ void addHitable(Hitable *hittable);
        __device__ void setPointLights(Vec3 *pointLights, int light_count);
        __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
        __device__ void empty();

        __device__ Vec3 handlePhong(const HitRecord &rec, Camera **cam) const;
        
        __device__ void debug_print() const;

        
        int hitable_count;

        Hitable **hitables;
        int hitable_capacity;
        Vec3 *pointLights;
        int point_light_count;
};


#endif