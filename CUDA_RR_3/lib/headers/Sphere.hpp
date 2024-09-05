// this is the class of a sphere object (child of the hitable class)


#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittable.hpp"
#include "RTWeekend.hpp"
#include "material.hpp"

class sphere : public hittable {
  public:
    __device__ sphere(const point3& center, double radius, material* mat) : center(center), radius(0 > radius ? 0 : radius), mat(mat) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

  private:
    point3 center;
    double radius;
    material* mat;
};

#endif