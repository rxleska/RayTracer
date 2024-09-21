#ifndef RTIAW_HPP
#define RTIAW_HPP

#include "../hittable/headers/Hitable.hpp"
#include "../hittable/headers/HitRecord.hpp"
#include "../hittable/headers/Octree.hpp"
#include "../hittable/headers/Sphere.hpp"
#include "../hittable/headers/Polygon.hpp"
#include "../hittable/headers/Octree.hpp"
#include "../materials/headers/Material.hpp"
#include "../materials/headers/Lambertian.hpp"
#include "../materials/headers/Metal.hpp"
#include "../materials/headers/Dielectric.hpp"
#include "../materials/headers/Light.hpp"
#include "../materials/headers/LambertianBordered.hpp"
#include "../materials/headers/Textured.hpp"
#include "../processing/headers/Camera.hpp"
#include "../processing/headers/Ray.hpp"
#include "../processing/headers/Vec3.hpp"


__device__ void create_RTIAW_sample(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 ***textures, int num_textures) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        // device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,
        //                        new Lambertian(Vec3(0.5, 0.5, 0.5)));


        device_object_list[i++] = Quad(Vec3(-500, 0, -500), Vec3(-500, 0, 500), Vec3(500, 0, 500), Vec3(500, 0, -500), new Lambertian(Vec3(0.5,0.5,0.5)));


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
        // device_object_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
        device_object_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Textured((*textures)[0],474,327));
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

        Vec3 lookfrom(13,2,3);

        *rand_state = local_rand_state;
        *d_world  = new Octree(device_object_list, i);
        ((Octree*)*d_world)->max_depth = 10;
        ((Octree*)*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);


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
        (*d_camera)->samples = 200;
        (*d_camera)->bounces = 100;
    }
}

#endif