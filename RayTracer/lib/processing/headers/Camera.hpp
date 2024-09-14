#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vec3.hpp"
#include "Ray.hpp"

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
        float ambient_light_level = 0.0f;
        int samples = 1;
        int bounces = 50;
        int x_res = 800;
        int y_res = 800;
        int msaa_x = 4;
    private:
        __device__ Vec3 random_in_unit_disk(curandState *state);
};

#endif 