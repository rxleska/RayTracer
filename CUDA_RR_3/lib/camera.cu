#include "headers/camera.hpp"
#include "headers/interval.hpp"

__device__ camera::camera() : aspect_ratio(16.0 / 9.0), image_width(1920), samples_per_pixel(1), max_depth(10), vfov(90), lookfrom(0, 0, 0), lookat(0, 0, -1), vup(0, 1, 0), defocus_angle(0), focus_dist(10.0) {}

__device__ void camera::initialize()
{
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    center = lookfrom;

    // Determine viewport dimensions.
    // auto focal_length = (lookfrom - lookat).length();
    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta / 2);
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width = viewport_height * (double(image_width) / image_height);

    // Calculate the camera basis vectors.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = viewport_width * u;
    auto viewport_v = viewport_height * -v;

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = center - (focus_dist * w) - 0.5 * (viewport_u + viewport_v);
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate camera defocus disk radii.
    auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = defocus_radius * u;
    defocus_disk_v = defocus_radius * v;

    pixel_sample_scale = 1.0 / samples_per_pixel;
}

__device__ ray camera::get_ray(int i, int j) const
{
    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel location i, j.

    auto offset = sample_square();
    auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__device__ vec3 camera::sample_square() const
{
    // TODO 16 will be wrong if the # thread vertically changes
    //get curand state
    curandState local_rand_state = d_rand_state[threadIdx.x + threadIdx.y * 16];
    return vec3(curand_uniform(&local_rand_state) - 0.5, curand_uniform(&local_rand_state) - 0.5, 0);

    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    // return vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

__device__ point3 camera::defocus_disk_sample() const
{
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__device__ color camera::ray_color(const ray &r, int depth, const hittable &world) const
{
    if (depth <= 0)
    {
        return color(0, 0, 0);
    }

    hit_record rec;

    if (world.hit(r, interval(0.001, infinity), rec))
    {
        ray scattered;
        color attenuation;
        int result = rec.mat->scatter(r, rec, attenuation, scattered);
        if (result == 1)
            return attenuation * ray_color(scattered, depth - 1, world);
        else if (result == 2)
            // hit a light source
            return ((light *)(rec.mat))->emitted();
        else
            return color(0, 0, 0);
    }

    // If no object was hit, return the background color. (Ambient light)
    // currently the ambient light is a function that makes a sky gradient (blue to white)
    double ambient_light_volume = 1;
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return ambient_light_volume * ((1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
}


__device__ void render_point(const hittable& world, int row, int column, color** image){

}