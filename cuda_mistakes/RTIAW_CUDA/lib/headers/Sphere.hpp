// this is the class of a sphere object (child of the hitable class)


#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "interval.hpp"
#include "ray.hpp"

// class sphere : public hittable {
class sphere{
  public:
    // __device__ sphere(const point3& center, double radius, shared_ptr<material> mat) : center(center), radius(std::fmax(0,radius)), mat(mat) {}
    __device__ sphere(const point3& center, double radius) : center(center), radius(0 > radius ? 0 : radius) {}

    // __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
    __device__ bool hit(const ray &r, interval ray_t) const;

  private:
    point3 center;
    double radius;
};

#endif