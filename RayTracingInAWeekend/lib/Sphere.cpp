// this is the class of a sphere object (child of the hitable class)

#include "headers/Sphere.hpp"

// hit function for sphere class
bool sphere::hit(const ray &r, interval ray_t, hit_record &rec) const
{
  // Calculate the ray from the center of the sphere to the origin of the ray
  vec3 oc = center - r.origin();

  // Calculate the coefficients of the quadratic equation
  float a = r.direction().length_squared();        // x^2 power  
  float h = dot(r.direction(), oc);                // x power (note: given the calulation of h, h = b/2 in the quadratic equation)
  float c = oc.length_squared() - radius * radius; // constant

  // Calculate the discriminant of the quadratic equation if it is negative the ray does not hit the sphere (i.e no real roots)
  float discriminant = h * h - a * c;
  if (discriminant < 0)
    return false;

  // Calculate the roots of the quadratic equation
  float sqrtd = std::sqrt(discriminant);
  float root = (h - sqrtd) / a;

  // Find the nearest root that lies in the acceptable range.
  if (!ray_t.surrounds(root))
  {
    root = (h + sqrtd) / a;
    if (!ray_t.surrounds(root))
      return false;
  }

  // Set the hit record and return true
  rec.t = root; //time of intersection
  rec.p = r.at(rec.t); //point of intersection
  vec3 outward_normal = (rec.p - center) / radius; //normal vector away from sphere on hit point
  rec.set_face_normal(r, outward_normal); //set the normal vector
  rec.mat = mat; //set the material
  return true;
}