#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray{
    public: 
        __device__ ray();
        __device__ ray(const vec3& origin, const vec3& direction);
        __device__ vec3 origin() const;
        __device__ vec3 direction() const;
        __device__ vec3 at(double t) const;
    private:
        vec3 orig;
        vec3 dir;
};


#endif