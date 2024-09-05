#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.hpp"
#include "RTWeekend.hpp"
#include "ArrayList.hpp"
#include "tppm.hpp"
#include "material.hpp"

#include <thread>

class camera {
  public:
    /* Public Camera Parameters Here */
    double aspect_ratio;
    int image_width;
    int samples_per_pixel;
    int max_depth;

    double vfov;
    point3 lookfrom;
    point3 lookat;
    vec3 vup;

    double defocus_angle;
    double focus_dist;

    __host__ __device__ camera();
    __host__ void render(const hittable& world);
    __host__ void render_mt(const hittable& world);
    __host__ void render_line(const hittable& world, int j, color ** image);
    __host__ void render_cuda(const hittable& world);
   
    __host__ __device__ ray get_ray(int i, int j) const;
    __host__ __device__ color ray_color(const ray& r, int depth, const hittable& world) const;

  private:
    /* Private Camera Variables Here */
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    double pixel_sample_scale; // Scale factor for pixel samples
    vec3   u, v, w;       // Camera basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    __host__ __device__ void initialize();
    __host__ __device__ vec3 sample_square() const;
    __host__ __device__ point3 defocus_disk_sample() const;
};

 __global__ void render_cuda_call(const hittable& world, color ** image, int image_width, int image_height, int samples_per_pixel, int max_depth, double pixel_sample_scale, camera *cam); 
 __global__ void initcurand(int image_width, int image_height);

#endif