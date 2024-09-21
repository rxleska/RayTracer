// hittable
#include "lib/hittable/headers/Hitable.hpp"
#include "lib/hittable/headers/HitRecord.hpp"
#include "lib/hittable/headers/Octree.hpp"
#include "lib/hittable/headers/Sphere.hpp"
#include "lib/hittable/headers/Polygon.hpp"
#include "lib/hittable/headers/Octree.hpp"
// materials
#include "lib/materials/headers/Material.hpp"
#include "lib/materials/headers/Lambertian.hpp"
#include "lib/materials/headers/Metal.hpp"
#include "lib/materials/headers/Dielectric.hpp"
#include "lib/materials/headers/Light.hpp"
#include "lib/materials/headers/LambertianBordered.hpp"
#include "lib/materials/headers/Textured.hpp"
// processing
#include "lib/processing/headers/Camera.hpp"
#include "lib/processing/headers/Ray.hpp"
#include "lib/processing/headers/Vec3.hpp"

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <fstream>


#define MAX_OBJECTS 500

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

__global__ void rand_init_singleton(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void init_texture(Vec3 ***textures, float *texture, int width, int height, int texture_index) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*textures)[texture_index] = new Vec3[width * height];
        for (int i = 0; i < width * height; i++) {
            (*textures)[texture_index][i] = Vec3(texture[i*3], texture[i*3+1], texture[i*3+2]);
        }
    }
}

__device__ Vec3 getColor(const Ray &r, Camera **cam, Scene **world, curandState *local_rand_state, bool &edge_hit) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0,1.0,1.0);
    for(int i = 0; i < (*cam)->bounces; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            int did_scatter = rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state);
            edge_hit = rec.edge_hit;
            if(did_scatter == 1) {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else if(did_scatter == 2){ //light hit return color
                return cur_attenuation * attenuation;
            }
            else {
                return Vec3(0.0,0.0,1.0);
            }
        }
        else {
            float ambient = (*cam)->ambient_light_level;
            Vec3 unit_direction = (cur_ray.direction).normalized();
            float t = 0.5f*(unit_direction.y + 1.0f);
            Vec3 c = Vec3(1.0, 1.0, 1.0)*(1.0f-t) + Vec3(0.5, 0.7, 1.0)*t;
            return cur_attenuation * (c*ambient);
        }
    }
    return Vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void free_world(Hitable **device_object_list, Scene **d_world, Camera **d_camera) {
    for(int i=0; i < (*d_world)->hitable_count; i++) {
        delete ((Sphere *)device_object_list[i])->mat;
        delete device_object_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void rand_init_render(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ float clamp(float x, float min, float max) {
    if(x < min) return min;
    if(x > max) return max;
    return x;
}

__global__ void render(uint8_t *fb, int max_x, int max_y, int ns, Camera **cam, Scene **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = (max_y - j - 1)*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0,0,0);
    bool edge_hit = false;
    bool edge_hit_check = false;
    int samples = (*cam)->samples;
    for(int s=0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col = col + getColor(r, cam, world, &local_rand_state, edge_hit_check);

        if(!edge_hit && edge_hit_check) {
            edge_hit = true;
            samples = samples * (*cam)->msaa_x;
        }
    }
    rand_state[pixel_index] = local_rand_state;
    col = col / float(samples);
    

    fb[pixel_index*3+0] = uint8_t(int(255.99*clamp(sqrt(col.x), 0.0f, 1.0f)));
    fb[pixel_index*3+1] = uint8_t(int(255.99*clamp(sqrt(col.y), 0.0f, 1.0f)));
    fb[pixel_index*3+2] = uint8_t(int(255.99*clamp(sqrt(col.z), 0.0f, 1.0f)));
}

#define RND (curand_uniform(&local_rand_state))


#include "lib/Scenes/TestScene.hpp"
#include "lib/Scenes/RTIAW.hpp"
#include "lib/Scenes/CornellBox.hpp"
#include "lib/Scenes/CornellRoomOfMirrors.hpp"

__global__ void create_world(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 ***textures, int num_textures){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // create_RTIAW_sample(device_object_list, d_world, d_camera, nx, ny, rand_state, textures, num_textures);
        create_test_scene(device_object_list, d_world, d_camera, nx, ny, rand_state, textures, num_textures);
        // create_Cornell_Box_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_Cornell_Box_Octree_ROM(device_object_list, d_world, d_camera, nx, ny, rand_state);
    }
}


int main() {
    //increase stack size
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    int nx = 512;
    // int nx = 500*1;
    // int nx = 1024;
    int ny = 512;
    // int ny = 500*1;
    // int ny = 1024;
    int ns = 100;
    // int tx = 20;
    // int ty = 12;
    // int tx = 16;
    // int ty = 10;
    int tx = 512;
    int ty = 1;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    // size_t fb_size = num_pixels*sizeof(vec3);
    size_t fb_size = num_pixels*sizeof(uint8_t)*3;

    // allocate Frame Buffer (fb)
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // timing
    clock_t start, stop;
    start = clock();


    // allocate random state for each pixel
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    // initialize random state for Octree generation
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init_singleton<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //LOAD IMAGES
    int num_textures = 1;
    Vec3 ***textures; //array of texture arrays
    checkCudaErrors(cudaMallocManaged((void **)&textures, num_textures*sizeof(Vec3 **)));

    //load image
    // int im1_w, im1_h;
    // im1_w = 474;
    // im1_h = 327;
    // float *im1 = load_texture("ExampleImage.ppm", im1_w, im1_h);
    // //copy to device
    // float *d_im1;
    // checkCudaErrors(cudaMalloc((void **)&d_im1, im1_w*im1_h*3*sizeof(float)));
    // checkCudaErrors(cudaMemcpy(d_im1, im1, im1_w*im1_h*3*sizeof(float), cudaMemcpyHostToDevice));
    // init_texture<<<1,1>>>(textures, d_im1, im1_w, im1_h, 0);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());




    // make our world of hitables & the camera
    Hitable **device_object_list;
    // int num_hitables = 22*22+1+3;
    int num_hitables = MAX_OBJECTS;
    checkCudaErrors(cudaMalloc((void **)&device_object_list, num_hitables*sizeof(Hitable *)));
    Scene **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Octree *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1,1>>>(device_object_list, d_world, d_camera, nx, ny, d_rand_state2, textures, num_textures);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // print world created 
    printf("World created\n");
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    rand_init_render<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    start = clock();

    //open file
    FILE *f = fopen("image.ppm", "wb");
    fprintf(f, "P6 %d %d 255\n", nx, ny);
    uint8_t *fb2 = (uint8_t *)malloc(fb_size*3);
    //direct memory copy
    checkCudaErrors(cudaMemcpy(fb2, fb, fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    fwrite(fb2, sizeof(uint8_t), 3*nx*ny, f);
    fclose(f);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "output took " << timer_seconds << " seconds.\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(device_object_list, d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(device_object_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}