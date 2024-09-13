#include "headers/Scene.hpp"

#include "headers/Hitable.hpp"
#include "../processing/headers/Ray.hpp"

__device__ Scene::Scene(){
    hitable_count = 0;
    hitable_capacity = 10;
    // hitables = new Hitable*[hitable_capacity]; // c++ new operator not supported in cuda
    hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_capacity);
}

__host__ void Scene::free_memory(){
    cudaFree(hitables);
}

__device__ void Scene::resize(int new_capacity){
    // Hitable **new_hitables = new Hitable*[new_capacity];
    Hitable **new_hitables = (Hitable**)malloc(sizeof(Hitable*) * new_capacity);
    for (int i = 0; i < hitable_count; i++) {
        new_hitables[i] = hitables[i];
    }
    // delete[] hitables;
    cudaFree(hitables);
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
    HitRecord current_hit;
    bool has_hit = false;
    float closest = t_max;
    for (int i = 0; i < hitable_count; i++) {
        if (hitables[i]->hit(ray, t_min, closest, current_hit)) {
            has_hit = true;
            closest = current_hit.t;
            rec = current_hit;
        }
    }

    return has_hit;
}

__device__ Scene::Scene(Hitable **hitables, int hitable_count){
    this->hitable_count = hitable_count;
    this->hitable_capacity = hitable_count;
    this->hitables = hitables;

}
