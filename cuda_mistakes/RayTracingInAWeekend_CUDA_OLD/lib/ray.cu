// basic ray class

#include "headers/ray.hpp"

__device__ point3 ray::origin() const { return orig; }
__device__ vec3 ray::direction() const { return dir; }

__device__ point3 ray::at(double t) const
{
    return orig + t * dir;
}
