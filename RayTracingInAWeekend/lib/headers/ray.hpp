// basic ray class

#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray {
    private:
        point3 orig;
        vec3 dir;
    
    public: 
        ray() {}
        ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        point3 origin() const;
        vec3 direction() const;

        point3 at(double t) const;
};


#endif