
#include "headers/Camera.hpp"

__device__ Camera::Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
    lens_radius = aperture / 2.0f;
    float theta = vfov*((float)M_PI)/180.0f;
    float half_height = tan(theta/2.0f);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = (lookfrom - lookat).normalized();
    u = (vup.cross(w)).normalized();
    v = w.cross(u);
    lower_left_corner = origin  - u*half_width*focus_dist - v*half_height*focus_dist - w*focus_dist;
    horizontal = u*2.0f*half_width*focus_dist;
    vertical = v*2.0f*half_height*focus_dist;
}


__device__ Ray Camera::get_ray(float s, float t, curandState *state) {
    Vec3 rd = random_in_unit_disk(state) * lens_radius;
    Vec3 offset = (u * rd.x) + (v * rd.y);
    return Ray(origin + offset, lower_left_corner + horizontal*s + vertical*t - origin - offset);
}


__device__ Vec3 Camera::random_in_unit_disk(curandState *state) {
    Vec3 p;
    do {
        p = Vec3(curand_uniform(state),curand_uniform(state),0) * 2.0f - Vec3(1,1,0);
    } while (p.dot(p) >= 1.0f);
    return p;
}