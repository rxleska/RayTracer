

#include "lib/headers/RTWeekend.hpp"
#include "lib/headers/tppm.hpp"
#include <vector> // arraylist, using to store objects in the image
#include "lib/headers/hittable.hpp"
#include "lib/headers/hittable_list.hpp"
#include "lib/headers/sphere.hpp"
#include "lib/headers/camera.hpp"
#include "lib/headers/Global.hpp"

#define WIDTH 1024
#define HEIGHT 768 
#define THREADS_X 16
#define THREADS_Y 16

#include <iostream>

// color ray_color(const ray& r, const hittable& world){
//     hit_record rec;
//     if(world.hit(r, interval(0, infinity), rec)){
//         return 0.5*color(rec.normal + color(1,1,1));
//     }


//     vec3 unit_direction = unit_vector(r.direction());
//     auto a = 0.5*(unit_direction.y() + 1.0);
//     return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
// }

//create the world of object in gpu memory
__global__ void create_world(hittable ** things, hittable_list ** world){
    //only construct on the first thread (its shared memory so all threads will have the same world)
    if(threadIdx.x == 0 && blockIdx.x == 0){

        // material * ground_material;
        // cudaMalloc(&ground_material, sizeof(lambertian));
        // memcpy(ground_material, &lambertian(color(0.5, 0.5, 0.5)), sizeof(lambertian));
        material* ground_material = new lambertian(color(0.5, 0.5, 0.5));
        *(things) = new sphere(point3(0,-1000,0), 1000, ground_material);

        // for (int a = -11; a < 11; a++) {
        //     for (int b = -11; b < 11; b++) {
        //         auto choose_mat = random_double();
        //         point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

        //         if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        //             material* sphere_material;

        //             if (choose_mat < 0.8) {
        //                 // diffuse
        //                 auto albedo = color::random() * color::random();
        //                 sphere_material = new lambertian(albedo);
        //                 world->add(new sphere(center, 0.2, sphere_material));
        //             } else if (choose_mat < 0.95) {
        //                 // metal
        //                 auto albedo = color::random(0.5, 1);
        //                 auto fuzz = random_double(0, 0.5);
        //                 sphere_material = new metal(albedo, fuzz);
        //                 world->add(new sphere(center, 0.2, sphere_material));
        //             } else {
        //                 // glass
        //                 sphere_material = new dielectric(1.5);
        //                 world->add(new sphere(center, 0.2, sphere_material));
        //             }
        //         }
        //     }
        // }

        material * material1 = new dielectric(1.5);
        *(things + 1) = new sphere(point3(0, 1, 0), 1.0, material1);

        *world = new hittable_list(things, 2);

        // material * material2 = new lambertian(color(0.4, 0.2, 0.1));
        // world->add(new sphere(point3(-4, 1, 0), 1.0, material2));

        // material * material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
        // world->add(new sphere(point3(4, 1, 0), 1.0, material3));

        // material * material4 = new light(color(1,1,1));
        // world->add(new sphere(point3(0,510,400), 500, material4));
    }
    else{
        return;
    }
}

__global__ void init_camera(camera *cam){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        // cam->aspect_ratio      = 16.0 / 9.0;
        cam->aspect_ratio      =  1.30279898; // 1024/768
        cam->image_width       = 768;
        cam->samples_per_pixel = 5;
        cam->max_depth         = 2;

        cam->vfov     = 20;
        cam->lookfrom = point3(13,2,3);
        cam->lookat   = point3(0,0,0);
        cam->vup      = vec3(0,1,0);

        cam->defocus_angle = 0.0;
        cam->focus_dist    = 1.0;

        cam->initialize();
    }
}


__global__ void render(hittable_list **world, camera *cam, color *img){
    //temp set all pixels red
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= WIDTH || y >= HEIGHT) return;

    cam->render_point(**world, x, y, img);

    // img[y*WIDTH + x] = color(1,0,0);
}

int main(int argc, char** argv){
    //world __device__ memory 
    hittable ** things;
    cudaMalloc(&things, 2*sizeof(hittable*));
    hittable_list ** world;
    cudaMalloc(&world, sizeof(hittable_list*));
    create_world<<<1,1>>>(things, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    camera *cam;
    cudaMalloc(&cam, sizeof(camera));
    init_camera<<<1,1>>>(cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    color *img;
    checkCudaErrors(cudaMallocManaged(&img, WIDTH*HEIGHT*sizeof(color)));

    curandState h_rand_state[THREADS_X * THREADS_Y];

    //create threads to render the image
    dim3 blocks(WIDTH/THREADS_X +1, HEIGHT/THREADS_Y + 1);
    dim3 threads(THREADS_X, THREADS_Y);
    render<<<blocks, threads>>>(world, cam, img);

    
    //wait for all threads to finish
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    return 0;
}