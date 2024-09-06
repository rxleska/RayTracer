
#ifndef POLYGON_HPP
#define POLYGON_HPP

#include "hittable.hpp"
#include "vec3.hpp"

class polygon : public hittable {
  public:
    polygon(const point3& v0, const point3& v1, const point3& v2, shared_ptr<material> mat) : vertices{v0, v1, v2}, mat(mat) {}

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

  private:
    point3 vertices[3];
    shared_ptr<material> mat;
};


#endif