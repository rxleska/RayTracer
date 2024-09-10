#include "headers/Ray.hpp"

Vec3 Ray::pointAt(float t) const {
    return origin + direction * t;
}