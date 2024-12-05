#ifndef TEST_SCENE_HPP
#define TEST_SCENE_HPP

#include "../hittable/headers/Hitable.hpp"
#include "../hittable/headers/HitRecord.hpp"
#include "../hittable/headers/Octree.hpp"
#include "../hittable/headers/Sphere.hpp"
#include "../hittable/headers/Polygon_T.hpp"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ void create_test_scene(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        // device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,new Lambertian(Vec3(0.7, 0.5, 0.5)));
        // device_object_list[0] = new Sphere(Vec3(0,-1000.0,-1), 1000,new Metal(Vec3(0.7f, 0.7f, 0.7f), 0.0));
        device_object_list[0] = Quad(Vec3(-20, 0, -20), Vec3(20, 0, -20), Vec3(20, 0, 20), Vec3(-20, 0, 20), new Lambertian(Vec3(0.9, 0.7, 0.7)));
        int i = 1;



        //sample sphere blue ball
        // device_object_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Lambertian(Vec3(0.1, 0.2, 0.5)));

        Material * text = new Textured(textures[0], 474, 266);
        // ((Textured*)text)->rot = 0.25f;
        device_object_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, text);



        Material * text1 = new Textured(textures[1], 474, 327);
        // // ((Textured*)text1)->rot = 0.25f;
        device_object_list[i++] = new Sphere(Vec3(-2, 1, 0), 1.0, text1);


        
        //test Polygon_T
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
        // device_object_list[i++] = new Polygon_T(vertices_poly, 4, new Lambertian(Vec3(0.9, 0.2, 0.1)));
        device_object_list[i++] = new Polygon_T(vertices_poly, 4, new Metal(Vec3(0.7f, 0.7f, 0.7f), 0.0));
        // device_object_list[i++] = new Polygon_T(vertices_poly, 4, text1, uvmap);


        //test mesh
        Material * red = new LambertianBordered(Vec3(0.9, 0.2, 0.1),Vec3(0.0,0.0,0.0));

        //rotate mesh about x axis by 90 degrees
        for(int j = 0; j < mesh_lengths[0]; j++) {
            float y = meshes[0][j].y;
            float z = meshes[0][j].z;
            meshes[0][j].y = y * cos(-M_PI/2) - z * sin(-M_PI/2);
            meshes[0][j].z = y * sin(-M_PI/2) + z * cos(-M_PI/2);
        }

        for(int j = 0; j < mesh_lengths[0]/3; j++) {
            device_object_list[i++] = Triangle(meshes[0][j*3], meshes[0][j*3 + 1], meshes[0][j*3 + 2], red);
        }
        

        // light source
        // Material * light = new Light(Vec3(1.0, 1.0, 1.0), 15.0);
        // device_object_list[i++] = new Sphere(Vec3(0, 510, 400), 200, light);

        //test Polygon_Ts
        // device_object_list[i++] = Triangle(vertices_poly[0], vertices_poly[1], vertices_poly[3], text1, uvmap[0], uvmap[1], uvmap[3]);
        // device_object_list[i++] = Triangle(vertices_poly[1], vertices_poly[2], vertices_poly[3], text1, uvmap[1], uvmap[2], uvmap[3]);

        
        //log Polygon_T vertice count
        // printf("Polygon_T vertices count: %d\n", ((Polygon_T *)device_object_list[i-1])->num_vertices);
        //log Polygon_T area
        // printf("Polygon_T area: %f\n", ((Polygon_T *)device_object_list[i-1])->area);
        //log normal
        // printf("Polygon_T normal: %f %f %f\n", ((Polygon_T *)device_object_list[i-1])->normal.x, ((Polygon_T *)device_object_list[i-1])->normal.y, ((Polygon_T *)device_object_list[i-1])->normal.z);
        
        Vec3 lookfrom(5,10,10);

        // printf("rand initing\n");
        *rand_state = local_rand_state;
        *d_world  = new Octree(device_object_list, i);
        ((Octree*)*d_world)->max_depth = 8;
        ((Octree*)*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);

        // printf("rand inited\n");
        // *d_world  = new Scene(device_object_list, i);

        Vec3 lookat(0,3,0);
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
        (*d_camera)->msaa_x = 4;
        (*d_camera)->samples = 1000;
        (*d_camera)->bounces = 64;
    }
}

#endif
