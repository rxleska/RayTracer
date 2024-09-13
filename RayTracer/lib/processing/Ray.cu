#include "headers/Ray.hpp"

__device__ Vec3 Ray::pointAt(float t) const {
    return origin + direction * t;
}