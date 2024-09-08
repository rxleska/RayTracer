// basic ray class

#include "headers/ray.hpp"

// returns the origin of the ray
point3 ray::origin() const { return orig; }

// returns the direction of the ray
vec3 ray::direction() const { return dir; }

// returns the point at a distance t from the origin of the ray
point3 ray::at(double t) const
{
    return orig + t * dir;
}
