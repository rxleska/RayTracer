#include "headers/camera.hpp"

#include <vector>

camera::camera() : aspect_ratio(16.0 / 9.0), image_width(1920), samples_per_pixel(1), max_depth(10), vfov(90), lookfrom(0, 0, 0), lookat(0, 0, -1), vup(0, 1, 0), defocus_angle(0), focus_dist(10.0) {}

void camera::count_thread() {
    thread_lock.lock();
    static int thread_count = 0;
    thread_count++;
    std::clog << "\rThreads finished: " << thread_count  << ' ' << std::flush;
    thread_lock.unlock();
}

void camera::count_pixels(int add){
    thread_lock.lock();
    static int pixel_count = 0;
    pixel_count += add;
    std::clog << "\rPixels rendered: " << pixel_count << ' ' << std::flush;
    thread_lock.unlock();
}

void camera::count_pixels_percent(int add){
    thread_lock.lock();
    static int pixel_count = 0;
    static float total_pixels = image_height * image_width;
    pixel_count += add;
    std::clog << "\rPixels rendered: " << float(pixel_count * 100)/ total_pixels << "%   " << ' ' << std::flush;
    thread_lock.unlock();
}

void camera::render(const hittable &world)
{
    initialize();

    color **image = new color *[image_height];
    for (int j = 0; j < image_height; j++)
    {
        image[image_height - j - 1] = new color[image_width];
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; sample++)
            {
                ray r = get_ray(i, j);
                pixel_color += ray_color(r, max_depth, world);
            }

            image[image_height - j - 1][i] = pixel_color * pixel_sample_scale;
        }
    }
    std::clog << "\nDone.\n";

    write_ppm("output.ppm", image, image_width, image_height);
}

void camera::render_mt(const hittable& world){
    initialize();

    color **image = new color *[image_height];
    for (int j = 0; j < image_height; j++)
    {
        image[image_height - j - 1] = new color[image_width];
    }

    // partitioning the pixels in to thread_count parts
    int partition_size = (image_width * image_height / thread_count) + 1;
    int start = 0;
    int end = partition_size;
    std::vector<std::thread> threads;
    for(; start < image_width * image_height; start += partition_size, end += partition_size){
        if(end > image_width * image_height){
            end = image_width * image_height;
        }
        // std::cout << "Starting thread with start: " << start << " and end: " << end << std::endl;
        threads.push_back(std::thread(&camera::render_mt_subset, this, std::ref(world), start, end, image));
    }

    for(auto &t : threads){
        t.join();
    }
    std::clog << "\nDone.\n";

    write_ppm("output.ppm", image, image_width, image_height);
}

void camera::render_mt_subset(const hittable& world, int start, int end, color ** image){
    int count = 0;
    int i;
    int j;
    for(int index = start; index < end; index++){
        j = index / image_width;
        i = index % image_width;
        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            ray r = get_ray(i, j);
            pixel_color += ray_color(r, max_depth, world);
        }

        image[image_height - j - 1][i] = pixel_color * pixel_sample_scale;

        count++;
        if(count % percent_modulo == 0){
            count_pixels_percent(percent_modulo);
        }
    }
    count_pixels_percent(count%percent_modulo);
}


void camera::render_mt_old(const hittable &world)
{
    initialize();

    std::vector<std::thread> threads;

    color **image = new color *[image_height];
    for (int j = 0; j < image_height; j++)
    {
        threads.push_back(std::thread(&camera::render_line, this, std::ref(world), j, image));
    }

    int i = 0;
    for (auto &t : threads)
    {
        // std::clog << "\rScanlines remaining: " << (image_height - i) << ' ' << std::flush;
        t.join();
        i++;
    }
    std::clog << "\nDone.\n";

    write_ppm("output.ppm", image, image_width, image_height);
}

void camera::render_line(const hittable &world, int j, color **image)
{
    image[image_height - j - 1] = new color[image_width];
    for (int i = 0; i < image_width; i++)
    {
        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            ray r = get_ray(i, j);
            pixel_color += ray_color(r, max_depth, world);
        }

        image[image_height - j - 1][i] = pixel_color * pixel_sample_scale;
    }
    count_thread();
}

void camera::initialize()
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

    percent_modulo = image_height *image_width / 200;
}

ray camera::get_ray(int i, int j) const
{
    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel location i, j.

    auto offset = sample_square();
    auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

vec3 camera::sample_square() const
{
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

point3 camera::defocus_disk_sample() const
{
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

color camera::ray_color(const ray &r, int depth, const hittable &world) const
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
            return dynamic_cast<light *>(rec.mat.get())->emitted();
        else
            return color(0, 0, 0);
    }

    // If no object was hit, return the background color. (Ambient light)
    // currently the ambient light is a function that makes a sky gradient (blue to white)
    double ambient_light_volume = 0.0;
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return ambient_light_volume * ((1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
}