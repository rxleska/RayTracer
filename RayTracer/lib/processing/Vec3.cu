#include "headers/Vec3.hpp"
#include <cmath>

__device__ void Vec3::normalize() {
    float m = mag();
    x /= m;
    y /= m;
    z /= m;
}

__device__ void Vec3::make_unit() {
    float mag = sqrt(x * x + y * y + z * z);
    x /= mag;
    y /= mag;
    z /= mag;
}

__device__ Vec3 Vec3::random(curandState *state) {
    return Vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
}

__device__ Vec3 Vec3::random(float min, float max, curandState *state) {
    return Vec3(min + (max - min) * curand_uniform(state), min + (max - min) * curand_uniform(state), min + (max - min) * curand_uniform(state));
}
