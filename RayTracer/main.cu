// hittable
#include "lib/hittable/headers/Hitable.hpp"
#include "lib/hittable/headers/HitRecord.hpp"
#include "lib/hittable/headers/Scene.hpp"
#include "lib/hittable/headers/Sphere.hpp"
// materials
#include "lib/materials/headers/Material.hpp"
#include "lib/materials/headers/Lambertian.hpp"
// processing
#include "lib/processing/headers/Camera.hpp"
#include "lib/processing/headers/Ray.hpp"
#include "lib/processing/headers/Vec3.hpp"

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <fstream>


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// __global__ void rand_init(curandState *rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curand_init(1984, 0, 0, rand_state);
//     }
// }

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__device__ Vec3 getColor(const Ray &r, Scene **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else {
                return Vec3(0.0,0.0,0.0);
            }
        }
        else {
            Vec3 unit_direction = (cur_ray.direction).normalized();
            float t = 0.5f*(unit_direction.y + 1.0f);
            Vec3 c = Vec3(1.0, 1.0, 1.0)*(1.0f-t) + Vec3(0.5, 0.7, 1.0)*t;
            return cur_attenuation * c;
        }
    }
    return Vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void free_world(Hitable **d_list, Scene **d_world, Camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((Sphere *)d_list[i])->mat;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(uint8_t *fb, int max_x, int max_y, int ns, Camera **cam, Scene **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col = col + getColor(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col = col * (1.0f/float(ns));
    col.x = sqrt(col.x);
    col.y = sqrt(col.y);
    col.z = sqrt(col.z);
    fb[pixel_index*3+0] = uint8_t(int(255.99*col.x));
    fb[pixel_index*3+1] = uint8_t(int(255.99*col.y));
    fb[pixel_index*3+2] = uint8_t(int(255.99*col.z));
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hitable **d_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,
                               new Lambertian(Vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a+RND,0.2,b+RND);
                d_list[i++] = new Sphere(center, 0.2,new Lambertian(Vec3(RND*RND, RND*RND, RND*RND)));
                // if(choose_mat < 0.8f) {
                //     d_list[i++] = new Sphere(center, 0.2,
                //                              new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                // }
                // else if(choose_mat < 0.95f) {
                //     d_list[i++] = new Sphere(center, 0.2,
                //                              new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                // }
                // else {
                //     d_list[i++] = new Sphere(center, 0.2, new dielectric(1.5));
                // }
            }
        }
        // d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
        // d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new Scene(d_list, 22*22+1+3);

        Vec3 lookfrom(13,2,3);
        Vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}


int main() {
    int nx = 1920/2;
    int ny = 1080/2;
    int ns = 25;
    int tx = 20;
    int ty = 12;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    // size_t fb_size = num_pixels*sizeof(vec3);
    size_t fb_size = num_pixels*sizeof(uint8_t)*3;

    // allocate FB
    // vec3 *fb;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    Hitable **d_list;
    // int num_hitables = 22*22+1+3;
    int num_hitables = 22*22+1+1;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(Hitable *)));
    Scene **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Scene *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    start = clock();

    //open file
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P6 %d %d 255\n", nx, ny);
    uint8_t *fb2 = (uint8_t *)malloc(fb_size*3);
    //direct memory copy
    checkCudaErrors(cudaMemcpy(fb2, fb, fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    fwrite(fb2, sizeof(uint8_t), 3*nx*ny, f);
    fclose(f);

    // Output FB as Image
    // std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    // for (int j = ny-1; j >= 0; j--) {
    //     for (int i = 0; i < nx; i++) {
    //         size_t pixel_index = j*nx + i;
    //         int ir = int(fb[pixel_index*3 + 0]);
    //         int ig = int(fb[pixel_index*3 + 1]);
    //         int ib = int(fb[pixel_index*3 + 2]);
    //         std::cout << ir << " " << ig << " " << ib << "\n";
    //     }
    // }
    // Output FB as Image (uint8_t) P6 PPM

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "output took " << timer_seconds << " seconds.\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}