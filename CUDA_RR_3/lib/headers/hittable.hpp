#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "RTWeekend.hpp"

class material; // Forward declaration

class hit_record {
  public:
    point3 p;
    vec3 normal;
    material* mat;
    double t;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal);
};

class hittable {
  public:
    __device__ virtual ~hittable() = default;

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
    // virtual bool hit(const ray& r, double ray_tmin, double ray_t, hit_record& rec) const = 0;
};

#endif