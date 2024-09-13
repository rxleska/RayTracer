#ifndef VEC3_HPP
#define VEC3_HPP

#include <curand_kernel.h>

class Vec3 {
public: 
    float x, y, z;

    __device__ Vec3(): x(0), y(0), z(0) {}
    __device__ Vec3(float x, float y, float z): x(x), y(y), z(z) {}

    __device__ inline float dot(const Vec3& v) const;
    __device__ inline Vec3 cross(const Vec3& v) const;
    __device__ inline float mag() const;
    __device__ inline float length() const;
    __device__ inline float mag2() const;
    __device__ inline float length2() const;
    __device__ inline Vec3 operator+(const Vec3& v) const;
    __device__ inline Vec3 operator-(const Vec3& v) const;
    __device__ inline Vec3 operator*(const Vec3& v) const;
    __device__ inline Vec3 operator*(const float& v) const;
    __device__ inline Vec3 operator/(const float& v) const;
    __device__ inline Vec3 operator*(const double& v) const;
    __device__ inline Vec3 operator/(const double& v) const;
    __device__ inline bool isZero() const;
    __device__ inline Vec3 normalized() const;
    __device__ void normalize();
    __device__ static Vec3 random(curandState *state);
    __device__ static Vec3 random(float min, float max, curandState *state);
    __device__ void make_unit();
};

#endif // VEC3_HPP
