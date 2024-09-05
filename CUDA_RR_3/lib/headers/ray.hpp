// basic ray class

#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray {
    private:
        point3 orig;
        vec3 dir;
    
    public: 
        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        __device__ point3 origin() const;
        __device__ vec3 direction() const;

        __device__ point3 at(double t) const;
};


#endif