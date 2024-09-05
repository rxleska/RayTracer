#include "headers/camera.hpp"

// #include <vector>
#include "headers/ArrayList.hpp"

__host__ __device__ camera::camera() : aspect_ratio(16.0 / 9.0), image_width(1920), samples_per_pixel(1), max_depth(10), vfov(90), lookfrom(0, 0, 0), lookat(0, 0, -1), vup(0, 1, 0), defocus_angle(0), focus_dist(10.0) {}

__host__ void camera::render(const hittable &world)
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

__host__ void camera::render_mt(const hittable &world)
{
    initialize();

    // std::vector<std::thread> threads;
    ArrayList<std::thread> threads;

    color **image = new color *[image_height];
    for (int j = 0; j < image_height; j++)
    {
        threads.add(std::thread(&camera::render_line, this, std::ref(world), j, image));
    }

    int i = 0;
    for (int j = 0; j < threads.size(); j++)
    {
        std::clog << "\rScanlines remaining: " << (image_height - i) << ' ' << std::flush;
        threads.get(j).join();
        i++;
    }
    std::clog << "\nDone.\n";

    write_ppm("output.ppm", image, image_width, image_height);
}

__host__ void camera::render_line(const hittable &world, int j, color **image)
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
}

__host__ void camera::render_cuda(const hittable& world){
    // Initialize the camera
    initialize();

    // Allocate memory for the image on the device
    color **d_image;
    cudaMalloc(&d_image, image_height * sizeof(color *));
    for (int i = 0; i < image_height; i++)
    {
        cudaMalloc(&d_image[i], image_width * sizeof(color));
    }

    //init curand
    initcurand<<<1, 1>>>(image_width, image_height);

    // Set up the kernel launch parameters
    dim3 blocks(image_width / 16 + 1, image_height / 16 + 1);
    dim3 threads(16, 16);

    // convert world to device 
    hittable *d_world;
    cudaMalloc(&d_world, sizeof(hittable));
    cudaMemcpy(d_world, &world, sizeof(hittable), cudaMemcpyHostToDevice);

    // Launch the kernel
    render_cuda_call<<<blocks, threads>>>(*d_world, d_image, image_width, image_height, samples_per_pixel, max_depth, pixel_sample_scale, this);

    // write the image to file
    color **image = new color *[image_height];
    cudaMemcpy(image, d_image, image_height * sizeof(color *), cudaMemcpyDeviceToHost);
    write_ppm("output.ppm", image, image_width, image_height);
}

__global__ void initcurand(int image_width, int image_height){
    d_rand_state = new curandState[image_width * image_height];
}

__global__ void render_cuda_call(const hittable& world, color ** image, int image_width, int image_height, int samples_per_pixel, int max_depth, double pixel_sample_scale, camera *cam){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < image_width && j < image_height){
        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            ray r = cam->get_ray(i, j);
            pixel_color += cam->ray_color(r, max_depth, world);
        }

        image[image_height - j - 1][i] = pixel_color * pixel_sample_scale;
    }
}

__host__ __device__ void camera::initialize()
{
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    center = lookfrom;

    // Determine viewport dimensions.
    // auto focal_length = (lookfrom - lookat).length();
    float theta = degrees_to_radians(vfov);
    float h = std::tan(theta / 2);
    float viewport_height = 2 * h * focus_dist;
    float viewport_width = viewport_height * (double(image_width) / image_height);

    // Calculate the camera basis vectors.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = viewport_width * u;
    vec3 viewport_v = viewport_height * -v;

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    vec3 viewport_upper_left = center - (focus_dist * w) - 0.5 * (viewport_u + viewport_v);
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate camera defocus disk radii.
    float defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = defocus_radius * u;
    defocus_disk_v = defocus_radius * v;

    pixel_sample_scale = 1.0 / samples_per_pixel;
}

__host__ __device__ ray camera::get_ray(int i, int j) const
{
    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel location i, j.

    vec3 offset = sample_square();
    vec3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

    vec3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    vec3 ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}



__host__ __device__ vec3 camera::sample_square() const
{
    #ifdef __CUDA_ARCH__
        // Device-specific code
        // get the random state
        curandState local_rand_state = d_rand_state[threadIdx.x + threadIdx.y*16];
        return vec3(curand_uniform(&local_rand_state)-0.5, curand_uniform(&local_rand_state)-0.5, 0);
    #else
        // Host-specific code
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    #endif
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
}

__host__ __device__ point3 camera::defocus_disk_sample() const
{
    // Returns a random point in the camera defocus disk.
    vec3 p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__host__ __device__ color camera::ray_color(const ray &r, int depth, const hittable &world) const
{
    if (depth <= 0)
    {
        return color(0, 0, 0);
    }

    hit_record rec;

    #ifdef __CUDA_ARCH__
        // Device-specific code
        if (world.hit(r, interval(0.001, d_infinity), rec))
    #else
        // Host-specific code
        if (world.hit(r, interval(0.001, infinity), rec))
    #endif
    {
        ray scattered;
        color attenuation;
        int result = rec.mat->scatter(r, rec, attenuation, scattered);
        if (result == 1)
            return attenuation * ray_color(scattered, depth - 1, world);
        else if (result == 2){
            // hit a light source
            if(rec.mat->getType() == LIGHT){
                //force cast to light
                return ((light *)rec.mat)->emitted();

            }
            else
                return color(0, 0, 0);
        }
        else
            return color(0, 0, 0);
    }

    // If no object was hit, return the background color. (Ambient light)
    // currently the ambient light is a function that makes a sky gradient (blue to white)
    double ambient_light_volume = 1;
    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return ambient_light_volume * ((1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
}