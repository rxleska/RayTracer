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

__device__ Vec3 getColor(const Ray &r, Camera **cam, Octree **world, curandState *local_rand_state, bool &edge_hit) {
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
                return Vec3(0.0,0.0,0.0);
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

__global__ void free_world(Hitable **device_object_list, Octree **d_world, Camera **d_camera) {
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

__global__ void render(uint8_t *fb, int max_x, int max_y, int ns, Camera **cam, Octree **world, curandState *rand_state) {
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


__device__ void create_test_Octree(Hitable **device_object_list, Octree **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,
                               new Lambertian(Vec3(0.7, 0.5, 0.5)));
        int i = 1;

        //sample sphere blue ball
        device_object_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Lambertian(Vec3(0.1, 0.2, 0.5)));

        //test polygon
        //cuda malloc a vertices array
        Vec3 * vertices_poly = new Vec3[4];
        vertices_poly[0] = Vec3( 0,  0, -5);
        vertices_poly[1] = Vec3(10,  0, -5);
        vertices_poly[2] = Vec3(10, 10, -5);
        vertices_poly[3] = Vec3( 0, 10, -5);
        // device_object_list[i++] = new Polygon(vertices_poly, 4, new Lambertian(Vec3(0.9, 0.2, 0.1)));
        device_object_list[i++] = new Polygon(vertices_poly, 4, new Metal(Vec3(0.7f, 0.5f, 0.3f), 0.0));
        

        
        //log polygon vertice count
        printf("Polygon vertices count: %d\n", ((Polygon *)device_object_list[i-1])->num_vertices);
        //log polygon area
        printf("Polygon area: %f\n", ((Polygon *)device_object_list[i-1])->area);
        //log normal
        printf("Polygon normal: %f %f %f\n", ((Polygon *)device_object_list[i-1])->normal.x, ((Polygon *)device_object_list[i-1])->normal.y, ((Polygon *)device_object_list[i-1])->normal.z);

        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        *d_world  = new Octree(device_object_list, i);
        //initialize Octree
        printf("Initializing Octree\n");
        (*d_world)->max_depth = 4;
        printf("Max depth set\n");
        (*d_world)->init();
        printf("Octree initialized\n");

        Vec3 lookfrom(0,5,10);
        Vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 60.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__device__ void create_RTIAW_sample(Hitable **device_object_list, Octree **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,
                               new Lambertian(Vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a+RND,0.2,b+RND);
                // device_object_list[i++] = new Sphere(center, 0.2,new Lambertian(Vec3(RND*RND, RND*RND, RND*RND)));
                if(choose_mat < 0.8f) {
                    device_object_list[i++] = new Sphere(center, 0.2,
                                             new Lambertian(Vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    device_object_list[i++] = new Sphere(center, 0.2,
                                             new Metal(Vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    device_object_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
        device_object_list[i++] = new Sphere(Vec3(0, 1,0),  1.0, new Dielectric(1.5));
        device_object_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
        device_object_list[i++] = new Sphere(Vec3(4, 1, 0),  1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

        //add sun light
        device_object_list[i++] = new Sphere(Vec3(0,510,400), 200, new Light(Vec3(1.0, 1.0, 1.0), 1.0));

        //test polygon
        //cuda malloc a vertices array
        Vec3 * vertices_poly = new Vec3[3];
        vertices_poly[0] = Vec3(0, 0, 0);
        vertices_poly[1] = Vec3(0, 10, 0);
        vertices_poly[2] = Vec3(10, 10, 0);
        device_object_list[i++] = new Polygon(vertices_poly, 3, new Lambertian(Vec3(0.9, 0.2, 0.1)));


        *rand_state = local_rand_state;
        *d_world  = new Octree(device_object_list, i);
        (*d_world)->max_depth = 4;
        (*d_world)->init();

        Vec3 lookfrom(13,2,3);
        Vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
        (*d_camera)->ambient_light_level = 0.8f;
        (*d_camera)->msaa_x = 4;
        (*d_camera)->samples = 500;
        (*d_camera)->bounces = 100;
    }
}

__device__ void create_Cornell_Box_Octree(Hitable **device_object_list, Octree **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        Material *white = new LambertianBordered(Vec3(1.0, 1.0, 1.0));
        Material *light = new Light(Vec3(1.0, 1.0, 1.0), 1.5);
        // Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15)*(1.0f/0.45f));
        // Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05)*(1.0f/0.65f));
        Material *green = new LambertianBordered(Vec3(0.12, 0.45, 0.15));
        Material *red = new LambertianBordered(Vec3(0.65, 0.05, 0.05));

        //floor 
        /*
        552.8 0.0   0.0   
        0.0 0.0   0.0
        0.0 0.0 559.2
        549.6 0.0 559.2
        */
        // device_object_list[i++] = Quad(Vec3(552.8, 0.0, 0.0),Vec3(0.0, 0.0, 0.0),Vec3(0.0, 0.0, 559.2),Vec3(552.8, 0.0, 559.2), white);
        device_object_list[i++] = Triangle(Vec3(552.8, 0.0, 0.0),Vec3(0.0, 0.0, 0.0),Vec3(0.0, 0.0, 559.2), white);
        device_object_list[i++] = Triangle(Vec3(552.8, 0.0, 0.0),Vec3(0.0, 0.0, 559.2),Vec3(552.8, 0.0, 559.2), white);
        

        //ceiling light 
        /*
        343.0 548.8 227.0 
        343.0 548.8 332.0
        213.0 548.8 332.0
        213.0 548.8 227.0
        */
        // device_object_list[i++] = Quad(Vec3(343.0, 548, 227.0),Vec3(343.0, 548, 332.0),Vec3(213.0, 548, 332.0),Vec3(213.0, 548, 227.0), light);
        device_object_list[i++] = Triangle(Vec3(343.0, 545, 227.0),Vec3(343.0, 545, 332.0),Vec3(213.0, 545, 332.0), light);
        device_object_list[i++] = Triangle(Vec3(343.0, 545, 227.0),Vec3(213.0, 545, 332.0),Vec3(213.0, 545, 227.0), light);


        //Ceiling
        /*
        556.0 548.8 0.0   
        556.0 548.8 559.2
        0.0 548.8 559.2
        0.0 548.8   0.0
        */
        // device_object_list[i++] = Quad(Vec3(556.0, 548.8, 0.0),Vec3(556.0, 548.8, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(0.0, 548.8, 0.0), white);
        device_object_list[i++] = Triangle(Vec3(556.0, 548.8, 0.0),Vec3(556.0, 548.8, 559.2),Vec3(0.0, 548.8, 559.2), white);
        device_object_list[i++] = Triangle(Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 559.2),Vec3(0.0, 548.8, 0.0), white);

        //back wall
        /*
        549.6   0.0 559.2 
        0.0   0.0 559.2
        0.0 548.8 559.2
        556.0 548.8 559.2
        */
        // device_object_list[i++] = Quad(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), white);

        //right wall
        /*
        0.0   0.0 559.2   
        0.0   0.0   0.0
        0.0 548.8   0.0
        0.0 548.8 559.2
        */
        // device_object_list[i++] = Quad(Vec3(0.0, 0.0, 559.2),Vec3(0.0, 0.0, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 548.8, 559.2), green);
        device_object_list[i++] = Triangle(Vec3(0.0, 0.0, 559.2),Vec3(0.0, 0.0, 0.0),Vec3(0.0, 548.8, 0.0), green);
        device_object_list[i++] = Triangle(Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 548.8, 559.2), green);

        //left wall
        /*
        552.8   0.0   0.0 
        549.6   0.0 559.2
        556.0 548.8 559.2
        556.0 548.8   0.0
        */
        // device_object_list[i++] = Quad(Vec3(552.8, 0.0, 0.0),Vec3(549.6, 0.0, 559.2),Vec3(556.0, 548.8, 559.2),Vec3(556.0, 548.8, 0.0), red);
        device_object_list[i++] = Triangle(Vec3(552.8, 0.0, 0.0),Vec3(549.6, 0.0, 559.2),Vec3(556.0, 548.8, 559.2), red);
        device_object_list[i++] = Triangle(Vec3(552.8, 0.0, 0.0),Vec3(556.0, 548.8, 559.2),Vec3(556.0, 548.8, 0.0), red);

        // //camera wall (we can see through this due to the directionality of polygons)
        // //uses white material
        // /*
        // 549.6   0.0 0 
        // 0.0   0.0 0
        // 0.0 548.8 0
        // 556.0 548.8 0
        // */
        // device_object_list[i++] = Quad(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0), white);

        //short block
        //uses white material
        //wall1
        /*
        130.0 165.0  65.0 
        82.0 165.0 225.0
        240.0 165.0 272.0
        290.0 165.0 114.0
        */
        // device_object_list[i++] = Quad(Vec3(130.0, 165.0, 65.0),Vec3(82.0, 165.0, 225.0),Vec3(240.0, 165.0, 272.0),Vec3(290.0, 165.0, 114.0), white);
        device_object_list[i++] = Triangle(Vec3(130.0, 165.0, 65.0),Vec3(82.0, 165.0, 225.0),Vec3(240.0, 165.0, 272.0), white);
        device_object_list[i++] = Triangle(Vec3(130.0, 165.0, 65.0),Vec3(240.0, 165.0, 272.0),Vec3(290.0, 165.0, 114.0), white);
        
        
        //wall2
        /*
        290.0   0.0 114.0
        290.0 165.0 114.0
        240.0 165.0 272.0
        240.0   0.0 272.0
        */
        //device_object_list[i++] = Quad(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
        device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0), white);
        device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
        
        
        //wall3
        /*
        130.0   0.0  65.0
        130.0 165.0  65.0
        290.0 165.0 114.0
        290.0   0.0 114.0
        */
        // device_object_list[i++] = Quad(Vec3(130.0, 0.0, 65.0),Vec3(130.0, 165.0, 65.0),Vec3(290.0, 165.0, 114.0),Vec3(290.0, 0.0, 114.0), white);
        device_object_list[i++] = Triangle(Vec3(130.0, 0.0, 65.0),Vec3(130.0, 165.0, 65.0),Vec3(290.0, 165.0, 114.0), white);
        device_object_list[i++] = Triangle(Vec3(130.0, 0.0, 65.0),Vec3(290.0, 165.0, 114.0),Vec3(290.0, 0.0, 114.0), white);
        
        
        //wall4
        /*
        82.0   0.0 225.0
        82.0 165.0 225.0
        130.0 165.0  65.0
        130.0   0.0  65.0
        */
        // device_object_list[i++] = Quad(Vec3(82.0, 0.0, 225.0),Vec3(82.0, 165.0, 225.0),Vec3(130.0, 165.0, 65.0),Vec3(130.0, 0.0, 65.0), white);
        device_object_list[i++] = Triangle(Vec3(82.0, 0.0, 225.0),Vec3(82.0, 165.0, 225.0),Vec3(130.0, 165.0, 65.0), white);
        device_object_list[i++] = Triangle(Vec3(82.0, 0.0, 225.0),Vec3(130.0, 165.0, 65.0),Vec3(130.0, 0.0, 65.0), white);
        
        
        //wall5
        /*
        240.0   0.0 272.0
        240.0 165.0 272.0
        82.0 165.0 225.0
        82.0   0.0 225.0
        */
        // device_object_list[i++] = Quad(Vec3(240.0, 0.0, 272.0),Vec3(240.0, 165.0, 272.0),Vec3(82.0, 165.0, 225.0),Vec3(82.0, 0.0, 225.0), white);
        device_object_list[i++] = Triangle(Vec3(240.0, 0.0, 272.0),Vec3(240.0, 165.0, 272.0),Vec3(82.0, 165.0, 225.0), white);
        device_object_list[i++] = Triangle(Vec3(240.0, 0.0, 272.0),Vec3(82.0, 165.0, 225.0),Vec3(82.0, 0.0, 225.0), white);

        

        //tall block
        //uses white material
        //wall1
        /*
        423.0 330.0 247.0
        265.0 330.0 296.0
        314.0 330.0 456.0
        472.0 330.0 406.0
        */
        // device_object_list[i++] = Quad(Vec3(423.0, 330.0, 247.0),Vec3(265.0, 330.0, 296.0),Vec3(314.0, 330.0, 456.0),Vec3(472.0, 330.0, 406.0), white);
        device_object_list[i++] = Triangle(Vec3(423.0, 330.0, 247.0),Vec3(265.0, 330.0, 296.0),Vec3(314.0, 330.0, 456.0), white);
        device_object_list[i++] = Triangle(Vec3(423.0, 330.0, 247.0),Vec3(314.0, 330.0, 456.0),Vec3(472.0, 330.0, 406.0), white);
        
        
        //wall2
        /*
        423.0   0.0 247.0
        423.0 330.0 247.0
        472.0 330.0 406.0
        472.0   0.0 406.0
        */
        // device_object_list[i++] = Quad(Vec3(423.0, 0.0, 247.0),Vec3(423.0, 330.0, 247.0),Vec3(472.0, 330.0, 406.0),Vec3(472.0, 0.0, 406.0), white);
        device_object_list[i++] = Triangle(Vec3(423.0, 0.0, 247.0),Vec3(423.0, 330.0, 247.0),Vec3(472.0, 330.0, 406.0), white);
        device_object_list[i++] = Triangle(Vec3(423.0, 0.0, 247.0),Vec3(472.0, 330.0, 406.0),Vec3(472.0, 0.0, 406.0), white);
        
        //wall3
        /*
        472.0   0.0 406.0
        472.0 330.0 406.0
        314.0 330.0 456.0
        314.0   0.0 456.0
        */
        // device_object_list[i++] = Quad(Vec3(472.0, 0.0, 406.0),Vec3(472.0, 330.0, 406.0),Vec3(314.0, 330.0, 456.0),Vec3(314.0, 0.0, 456.0), white);
        device_object_list[i++] = Triangle(Vec3(472.0, 0.0, 406.0),Vec3(472.0, 330.0, 406.0),Vec3(314.0, 330.0, 456.0), white);
        device_object_list[i++] = Triangle(Vec3(472.0, 0.0, 406.0),Vec3(314.0, 330.0, 456.0),Vec3(314.0, 0.0, 456.0), white);
        
        
        //wall4
        /*
        314.0   0.0 456.0
        314.0 330.0 456.0
        265.0 330.0 296.0
        265.0   0.0 296.0
        */
        // device_object_list[i++] = Quad(Vec3(314.0, 0.0, 456.0),Vec3(314.0, 330.0, 456.0),Vec3(265.0, 330.0, 296.0),Vec3(265.0, 0.0, 296.0), white);
        device_object_list[i++] = Triangle(Vec3(314.0, 0.0, 456.0),Vec3(314.0, 330.0, 456.0),Vec3(265.0, 330.0, 296.0), white);
        device_object_list[i++] = Triangle(Vec3(314.0, 0.0, 456.0),Vec3(265.0, 330.0, 296.0),Vec3(265.0, 0.0, 296.0), white);
        
        //wall5
        /*
        265.0   0.0 296.0
        265.0 330.0 296.0
        423.0 330.0 247.0
        423.0   0.0 247.0
        */
        // device_object_list[i++] = Quad(Vec3(265.0, 0.0, 296.0),Vec3(265.0, 330.0, 296.0),Vec3(423.0, 330.0, 247.0),Vec3(423.0, 0.0, 247.0), white);
        device_object_list[i++] = Triangle(Vec3(265.0, 0.0, 296.0),Vec3(265.0, 330.0, 296.0),Vec3(423.0, 330.0, 247.0), white);
        device_object_list[i++] = Triangle(Vec3(265.0, 0.0, 296.0),Vec3(423.0, 330.0, 247.0),Vec3(423.0, 0.0, 247.0), white);

        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        *d_world  = new Octree(device_object_list, i);
        //initialize Octree
        printf("Initializing Octree\n");
        (*d_world)->max_depth = 32;
        printf("Max depth set\n");
        (*d_world)->init();
        printf("Octree initialized\n");

        Vec3 lookfrom(278.0f, 278.0f, -400.0f);
        Vec3 lookat(278.0f, 278.0f, 0.0f);
        float dist_to_focus = 15.0; (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 70.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
        (*d_camera)->ambient_light_level = 0.0f;
        (*d_camera)->msaa_x = 2;
        (*d_camera)->samples = 64;
        (*d_camera)->bounces = 100;
    }
}


__global__ void create_world(Hitable **device_object_list, Octree **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        create_RTIAW_sample(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_test_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_Cornell_Box_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
    }
}


int main() {
    //increase stack size
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    // int nx = 1920/2;
    // int nx = 500*1;
    int nx = 1024;
    // int ny = 1080/2;
    // int ny = 500*1;
    int ny = 1024;
    int ns = 64;
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

    // make our world of hitables & the camera
    Hitable **device_object_list;
    // int num_hitables = 22*22+1+3;
    int num_hitables = MAX_OBJECTS;
    checkCudaErrors(cudaMalloc((void **)&device_object_list, num_hitables*sizeof(Hitable *)));
    Octree **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Octree *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1,1>>>(device_object_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // print world created 
    printf("World created\n");

    clock_t start, stop;
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
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
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