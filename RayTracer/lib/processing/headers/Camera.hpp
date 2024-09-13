#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vec3.hpp"
#include "Ray.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ Vec3 random_in_unit_disk(curandState *state) {
    Vec3 p;
    do {
        p = Vec3(curand_uniform(state),curand_uniform(state),0) * 2.0f - Vec3(1,1,0);
    } while (p.dot(p) >= 1.0f);
    return p;
}

class Camera{
    public:
        __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist);
        __device__ Ray get_ray(float s, float t, curandState *state);

        Vec3 origin;
        Vec3 lower_left_corner;
        Vec3 horizontal;
        Vec3 vertical;
        Vec3 u, v, w;
        float lens_radius;
};

#endif 