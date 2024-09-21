#ifndef TEST_SCENE_HPP
#define TEST_SCENE_HPP

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

__device__ void create_test_Octree(Hitable **device_object_list, Octree **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 ***textures, int num_textures) {
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
        
        Vec3 lookfrom(0,5,10);

        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        *d_world  = new Octree(device_object_list, i);
        //initialize Octree
        printf("Initializing Octree\n");
        (*d_world)->max_depth = 4;
        printf("Max depth set\n");
        (*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);
        printf("Octree initialized\n");

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

#endif