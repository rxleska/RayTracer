#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP


#include "hittable.hpp"
// #include <vector>
#include "ArrayList.hpp"

#include "RTWeekend.hpp"

class hittable_list : public hittable {
  public:
    ArrayList<hittable*> objects;

    __host__ __device__ hittable_list();
    __host__ __device__ hittable_list(hittable* object);

    __host__ __device__ void clear();

    __host__ __device__ void add(hittable* object);

    __host__ __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
};

#endif 
