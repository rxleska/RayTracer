#include "headers/hittable_list.hpp"

__device__ hittable_list::hittable_list() {}
__device__ hittable_list::hittable_list(hittable* object) { add(object); }

__device__ void hittable_list::clear() { objects.clear(); }

__device__ void hittable_list::add(hittable* object)
{
    objects.push_back(object);
}

__device__ bool hittable_list::hit(const ray &r, interval ray_t, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < objects.get_size(); i++)
    {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
