#ifndef VEC3_HPP
#define VEC3_HPP

#include <curand_kernel.h>

class Vec3
{
public:
    float x, y, z;

    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ inline float dot(const Vec3 &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    __device__ inline Vec3 cross(const Vec3 &v) const
    {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    __device__ inline float mag() const
    {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ inline float length() const
    {
        return mag();
    }

    __device__ inline float mag2() const
    {
        return x * x + y * y + z * z;
    }

    __device__ inline float length2() const
    {
        return mag2();
    }

    __device__ inline Vec3 operator+(const Vec3 &v) const
    {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __device__ inline Vec3 operator-(const Vec3 &v) const
    {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __device__ inline Vec3 operator*(const Vec3 &v) const
    {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __device__ inline Vec3 operator*(const float &v) const
    {
        return Vec3(x * v, y * v, z * v);
    }

    __device__ inline Vec3 operator/(const float &v) const
    {
        return Vec3(x / v, y / v, z / v);
    }

    __device__ inline Vec3 operator*(const double &v) const
    {
        return Vec3(x * v, y * v, z * v);
    }

    __device__ inline Vec3 operator/(const double &v) const
    {
        return Vec3(x / v, y / v, z / v);
    }

    __device__ inline bool isZero() const
    {
        return x > -0.00001 && x < 0.00001 && y > -0.00001 && y < 0.00001 && z > -0.00001 && z < 0.00001;
    }

    __device__ inline Vec3 normalized() const
    {
        float m = mag();
        return Vec3(x / m, y / m, z / m);
    }
    __device__ void normalize();
    __device__ static Vec3 random(curandState *state);
    __device__ static Vec3 random(float min, float max, curandState *state);
    __device__ void make_unit();

    __device__ static Vec3 random_on_hemisphere(curandState *state, const Vec3 &normal);
};

#endif // VEC3_HPP
