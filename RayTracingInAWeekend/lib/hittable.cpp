#include "headers/hittable.hpp"


// Sets the hit record normal vector.
// NOTE: the parameter `outward_normal` is assumed to have unit length.
void hit_record::set_face_normal(const ray &r, const vec3 &outward_normal)
{

  front_face = dot(r.direction(), outward_normal) < 0;
  normal = front_face ? outward_normal : -outward_normal;
}