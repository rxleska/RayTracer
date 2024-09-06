// this class is a simple 3D vector class (header file) (not defining class in a separate file)


#ifndef VEC3_HPP
#define VEC3_HPP

#include <iostream>
#include <cmath>
#include "Global.hpp"


// inline double random_double() {
//     // Returns a random real in [0,1).
//     // return std::rand() / (RAND_MAX + 1.0);
//     // usage of rand() is not allowed in CUDA
//     // using curand instead
// }

// inline double random_double(double min, double max) {
//     // Returns a random real in [min,max).
//     return min + (max-min)*random_double();
// }

class vec3 {
    public:
        double e[3];
        __host__ __device__ vec3();
        __host__ __device__ vec3(double e0, double e1, double e2);

        __host__ __device__ double x() const;
        __host__ __device__ double y() const;
        __host__ __device__ double z() const;

        __host__ __device__ vec3 operator-() const;
        __host__ __device__ double operator[](int i) const;
        __host__ __device__ double& operator[](int i);

        __host__ __device__ vec3& operator+=(const vec3 &v);
        __host__ __device__ vec3& operator*=(const double t);
        __host__ __device__ vec3& operator/=(const double t);

        __host__ __device__ double length() const;
        __host__ __device__ double length_squared() const;
        __host__ __device__ bool near_zero() const;

        __device__ static vec3 random()
        {
            //TODO 16 might be wrong if the thread height is not 16
            return vec3(0.5,0.5,0.5);
            // curandState *local_rand_state = &d_rand_state[threadIdx.x + threadIdx.y * 16];
            // return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
            // return vec3(random_double(), random_double(), random_double());
        }

        __device__ static vec3 random(double min, double max)
        {
            //TODO 16 might be wrong if the thread height is not 16
            // curandState *local_rand_state = &d_rand_state[threadIdx.x + threadIdx.y * 16];
            // return vec3(curand_uniform(local_rand_state) * (max - min) + min, curand_uniform(local_rand_state) * (max - min) + min, curand_uniform(local_rand_state) * (max - min) + min);

            return vec3((max-min)/2 + min,(max-min)/2 + min,(max-min)/2 + min);

            // return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
        }
};


using point3 = vec3; // 3D point (alias)

// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ inline vec3 random_in_unit_disk() {
    //get curandState
    //TODO 16 might be wrong if the thread height is not 16
    // curandState *local_rand_state = &d_rand_state[threadIdx.x + threadIdx.y * 16];
    return vec3(0.5,0.5,0);
    // while (true) {
    //     auto p = vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) * 2 - vec3(1, 1, 0);
    //     if (p.length_squared() < 1)
    //         return p;
    // }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif