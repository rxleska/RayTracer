// basic ray class

#include "headers/ray.hpp"

point3 ray::origin() const { return orig; }
vec3 ray::direction() const { return dir; }

point3 ray::at(double t) const
{
    return orig + t * dir;
}
