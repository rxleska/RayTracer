#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP


#include "hittable.hpp"
#include "DynamicArray.hpp"
#include "RTWeekend.hpp"

class hittable_list : public hittable {
  public:
    // DynamicArray<hittable*> objects;
    hittable** objects;
    int size;

    __device__ hittable_list();
    __device__ hittable_list::hittable_list(hittable** objects, int size);
    // __device__ hittable_list(hittable* object);

    __device__ void clear();

    // __device__ void add(hittable* object);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
};

#endif 
