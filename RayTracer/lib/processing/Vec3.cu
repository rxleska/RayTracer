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

#ifndef M_PI 
#define M_PI 3.14159265358979323846
#endif

__device__ Vec3 Vec3::random_on_hemisphere(curandState *state, const Vec3 &normal) {
    float h0 = curand_uniform(state);
    float h1 = curand_uniform(state);
    float theta = acos(h0); //between 0 and pi/2 since h0 is between 0 and 1
    float phi = 2 * M_PI * h1; // between 0 and 2pi

    //TODO CHECK THAT MY ROTATION OF z oriented hemisphere to normal is correct

    Vec3 hemisphere = Vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    
    Vec3 r = normal.cross(Vec3(0,0,1));
    double rcos = normal.dot(Vec3(0,0,1));
    rcos = rcos / (normal.mag());
    double rtheta = -acos(rcos);

    hemisphere = hemisphere * rcos + r * r.dot(hemisphere) * (1 - rcos) + r.cross(hemisphere) * sin(rtheta);

    return hemisphere;
}
