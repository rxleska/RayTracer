#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.hpp"
#include "RTWeekend.hpp"
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

    camera();

    void render(const hittable& world);
    void render_mt(const hittable& world);
    void render_line(const hittable& world, int j, color ** image);
    color ray_color(const ray& r, int depth, const hittable& world) const;

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

    void initialize();

    ray get_ray(int i, int j) const;
    vec3 sample_square() const;
    point3 defocus_disk_sample() const;
};

#endif