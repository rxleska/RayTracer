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

__device__ void create_test_scene(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        // device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,new Lambertian(Vec3(0.7, 0.5, 0.5)));
        device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,new Metal(Vec3(0.7f, 0.7f, 0.7f), 0.0));
        int i = 1;



        //sample sphere blue ball
        // device_object_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Lambertian(Vec3(0.1, 0.2, 0.5)));

        // Material * text = new Textured(textures[0], 474, 266);
        // ((Textured*)text)->rot = 0.25f;
        // device_object_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, text);



        Material * text1 = new Textured(textures[1], 474, 327);
        // ((Textured*)text1)->rot = 0.25f;
        // device_object_list[i++] = new Sphere(Vec3(-2, 1, 0), 1.0, text1);


        //test polygon
        //cuda malloc a vertices array
        Vec3 * vertices_poly = new Vec3[4];
        vertices_poly[0] = Vec3( -10,  0, -5);
        vertices_poly[1] = Vec3(10,  0, -5);
        vertices_poly[2] = Vec3(10, 10, -5);
        vertices_poly[3] = Vec3( -10, 10, -5);

        Vec3 * uvmap = new Vec3[4];
        uvmap[0] = Vec3(0, 0, 0);
        uvmap[1] = Vec3(1, 0, 0);
        uvmap[2] = Vec3(1, 1, 0);
        uvmap[3] = Vec3(0, 1, 0);
        // device_object_list[i++] = new Polygon(vertices_poly, 4, new Lambertian(Vec3(0.9, 0.2, 0.1)));
        // device_object_list[i++] = new Polygon(vertices_poly, 4, new Metal(Vec3(0.7f, 0.7f, 0.7f), 0.0));
        device_object_list[i++] = new Polygon(vertices_poly, 4, text1, uvmap);
        

        
        //log polygon vertice count
        // printf("Polygon vertices count: %d\n", ((Polygon *)device_object_list[i-1])->num_vertices);
        //log polygon area
        // printf("Polygon area: %f\n", ((Polygon *)device_object_list[i-1])->area);
        //log normal
        // printf("Polygon normal: %f %f %f\n", ((Polygon *)device_object_list[i-1])->normal.x, ((Polygon *)device_object_list[i-1])->normal.y, ((Polygon *)device_object_list[i-1])->normal.z);
        
        Vec3 lookfrom(3,2,10);

        // printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        // printf("rand inited\n");
        *d_world  = new Scene(device_object_list, i);

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
        (*d_camera)->ambient_light_level = 0.9f;
        (*d_camera)->msaa_x = 16;
        (*d_camera)->samples = 64;
        (*d_camera)->bounces = 50;
    }
}

#endif
