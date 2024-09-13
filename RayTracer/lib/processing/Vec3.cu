#include "headers/Vec3.hpp"
#include <cmath>

__device__ inline float Vec3::dot(const Vec3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

__device__ inline Vec3 Vec3::cross(const Vec3& v) const {
    return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

__device__ inline float Vec3::mag() const {
    return sqrt(x * x + y * y + z * z);
}

__device__ inline float Vec3::length() const {
    return mag();
}

__device__ inline float Vec3::mag2() const {
    return x * x + y * y + z * z;
}

__device__ inline float Vec3::length2() const {
    return mag2();
}

__device__ inline Vec3 Vec3::operator+(const Vec3& v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
}

__device__ inline Vec3 Vec3::operator-(const Vec3& v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
}

__device__ inline Vec3 Vec3::operator*(const Vec3& v) const {
    return Vec3(x * v.x, y * v.y, z * v.z);
}

__device__ inline Vec3 Vec3::operator*(const float& v) const {
    return Vec3(x * v, y * v, z * v);
}

__device__ inline Vec3 Vec3::operator/(const float& v) const {
    return Vec3(x / v, y / v, z / v);
}

__device__ inline Vec3 Vec3::operator*(const double& v) const {
    return Vec3(x * v, y * v, z * v);
}

__device__ inline Vec3 Vec3::operator/(const double& v) const {
    return Vec3(x / v, y / v, z / v);
}

__device__ inline bool Vec3::isZero() const {
    return x > -0.00001 && x < 0.00001 && y > -0.00001 && y < 0.00001 && z > -0.00001 && z < 0.00001;
}

__device__ inline Vec3 Vec3::normalized() const {
    float m = mag();
    return Vec3(x / m, y / m, z / m);
}

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
