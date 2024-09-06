// this class is a simple 3D vector class (header file) (not defining class in a separate file)

#include "headers/vec3.hpp"

__host__ __device__ vec3::vec3() : e{0, 0, 0} {}
__host__ __device__ vec3::vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

__host__ __device__ double vec3::x() const { return e[0]; }
__host__ __device__ double vec3::y() const { return e[1]; }
__host__ __device__ double vec3::z() const { return e[2]; }

__host__ __device__ vec3 vec3::operator-() const { return vec3(-e[0], -e[1], -e[2]); }
__host__ __device__ double vec3::operator[](int i) const { return e[i]; }
__host__ __device__ double &vec3::operator[](int i) { return e[i]; }

__host__ __device__ vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator*=(const double t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ vec3 &vec3::operator/=(const double t)
{
    return *this *= 1 / t;
}

__host__ __device__ double vec3::length() const
{
    return sqrt(length_squared());
}

__host__ __device__ double vec3::length_squared() const
{
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

__host__ __device__ bool vec3::near_zero() const
{
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
}

