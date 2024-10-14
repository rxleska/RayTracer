#include "headers/Scene.hpp"

#include "headers/Hitable.hpp"
#include "../processing/headers/Ray.hpp"

#include <iostream>


__device__ Scene::Scene(){
    hitable_count = 0;
    hitable_capacity = 2;
    // printf("Scene constructor\n");
    // hitables = new Hitable*[hitable_capacity]; // c++ new operator not supported in cuda
    hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_capacity);

    //point lights
    pointLights = nullptr;
    point_light_count = 0;
}

__host__ void Scene::free_memory(){
    cudaFree(hitables);
}

__device__ void Scene::resize(int new_capacity){
    // printf("Resizing\n");
    // Hitable **new_hitables = new Hitable*[new_capacity];
    Hitable **new_hitables = (Hitable**)malloc(sizeof(Hitable*) * new_capacity);
    for (int i = 0; i < hitable_count; i++) {
        new_hitables[i] = hitables[i];
    }
    // delete[] hitables;
    free(hitables);
    hitables = new_hitables;
    hitable_capacity = new_capacity;

}
__device__ void Scene::addHitable(Hitable *hittable){
    if (hitable_count == hitable_capacity) {
        resize(hitable_capacity * 2);
    }
    hitables[hitable_count++] = hittable;
}
__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const{
    bool has_hit = false;
    // float closest = t_max;
    for (int i = 0; i < hitable_count; i++) {
        if (hitables[i]->hit(ray, t_min, t_max, rec)) {
            has_hit = true;
            t_max = rec.t;
            // rec = current_hit;
        }
    }

    return has_hit;
}

__device__ Scene::Scene(Hitable **hitables, int hitable_count){
    this->hitable_count = hitable_count;
    this->hitable_capacity = hitable_count;
    this->hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_count);
    for (int i = 0; i < hitable_count; i++) {
        this->hitables[i] = hitables[i];
    }
}


__device__ void Scene::empty(){
    hitable_count = 0;
}


__device__ void Scene::debug_print() const{
    for(int i = 0; i < hitable_count; i++){
        hitables[i]->debug_print();
    }
}

__device__ void Scene::setPointLights(Vec3 *pointLights, int light_count){
    this->pointLights = (Vec3*)malloc(sizeof(Vec3) * light_count);
    for (int i = 0; i < light_count; i++) {
        this->pointLights[i] = pointLights[i];
    }
    this->point_light_count = light_count;
}
