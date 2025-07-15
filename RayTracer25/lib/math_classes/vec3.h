#ifndef VEC3_H
#define VEC3_H

#include "math.h" // using math.h over <cmath> due to better usage on the device (cmath works on the device but is not as efficient as it is on the host)
// #include <iostream> // we will only call this on the host

struct vec3 {
    float x;
    float y;
    float z;
};

__host__ __device__ inline vec3 new_vec3(float x, float y, float z) {
    return {x, y, z};
}

__host__ __device__ inline vec3& operator+=(vec3& self, const vec3& v) {
    self.x += v.x; self.y += v.y; self.z += v.z;
    return self;
}
__host__ __device__ inline vec3& operator-=(vec3& self, const vec3& v) {
    self.x -= v.x; self.y -= v.y; self.z -= v.z;
    return self;
}
__host__ __device__ inline vec3& operator*=(vec3& self, float t) {
    self.x *= t; self.y *= t; self.z *= t;
    return self;
}
__host__ __device__ inline vec3& operator/=(vec3& self, float t) {
    self.x /= t; self.y /= t; self.z /= t;
    return self;
}

__host__ __device__ inline float vec3_length(const vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline void vec3_normalize(vec3& v) {
    float len = vec3_length(v);
    if (len == 0) {
        v.x = 0; v.y = 0; v.z = 0; // Avoid division by zero
    } else {
        v /= len;
    }
}

// negation operator (unary minus)
__host__ __device__ inline vec3 operator-(const vec3&self)  {
    return {-self.x, -self.y, -self.z};
}

// definition operators (i.e. not member functions) 
__host__ __device__ inline vec3 operator+(const vec3& v, const vec3& u) {
    return {v.x + u.x, v.y + u.y, v.z + u.z};
}

__host__ __device__ inline vec3 operator-(const vec3& v, const vec3& u) {
    return {v.x - u.x, v.y - u.y, v.z - u.z};
}

// element-wise multiplication operator (NOTE: not dot product or cross product)
__host__ __device__ inline vec3 operator*(const vec3& v, const vec3& u) {
    return {v.x * u.x, v.y * u.y, v.z * u.z};
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return {v.x * t, v.y * t, v.z * t};
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return {v.x * t, v.y * t, v.z * t};
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return {v.x / t, v.y / t, v.z / t};
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
    return {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    };
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    float len = vec3_length(v);
    if (len == 0) return {0, 0, 0}; // Avoid division by zero
    return v / len;
}

#include "../random_cuda_funcs.h"

__device__ inline vec3 random_unit_vector(curandState* curandState) {
    // Generate a random vector in the unit sphere
    float theta = curand_uniform(curandState) * 2.0f * 3.1415927f; // Random angle
    float phi = acosf(2.0f * curand_uniform(curandState) - 1.0f); // Random angle for polar coordinates
    return {
        sinf(phi) * cosf(theta),
        sinf(phi) * sinf(theta),
        cosf(phi)
    };
}

__device__ inline bool is_near_zero(const vec3& v) {
    const float epsilon = 1e-8f; // A small value to compare against
    return (fabs(v.x) < epsilon && fabs(v.y) < epsilon && fabs(v.z) < epsilon);
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    // Reflect vector v around normal n
    return v - (2.0f * dot(v, n) * n);
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    // Refract vector uv through normal n with ratio etai_over_etat
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    // vec3 r_out_parallel = -sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp))) * n;
    vec3 r_out_parallel = -sqrtf(fmaxf(0.0f, 1.0f - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline float schlick(float cosine, float ref_idx) {
    // Schlick's approximation for reflectance
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    // TODO maybe look into polynomial approximation for this
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}


#endif // VEC3_H