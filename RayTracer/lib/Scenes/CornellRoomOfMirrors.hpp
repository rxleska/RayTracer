#ifndef CORNELL_ROOM_OF_MIRRORS_HPP
#define CORNELL_ROOM_OF_MIRRORS_HPP

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

__device__ void create_Cornell_Box_Octree_ROM(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        

        Material *white = new LambertianBordered(Vec3(1.0, 1.0, 1.0));
        Material *mirror = new Metal(Vec3(0.9, 0.9, 0.9), 0.001);
        Material *light = new Light(Vec3(1.0, 1.0, 1.0), 15.0);
        Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15)*(1.0f/0.45f));
        Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05)*(1.0f/0.65f));
        // Material *green = new LambertianBordered(Vec3(0.12, 0.45, 0.15));
        // Material *red = new LambertianBordered(Vec3(0.65, 0.05, 0.05));
        // Material *green = new Metal(Vec3(0.12, 0.45, 0.15), 0.001);
        // Material *red = new Metal(Vec3(0.65, 0.05, 0.05), 0.001);

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
        device_object_list[i++] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(343.0, 548.5, 332.0),Vec3(213.0, 548.5, 332.0), light);
        device_object_list[i++] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(213.0, 548.5, 332.0),Vec3(213.0, 548.5, 227.0), light);

        Hitable **lightList = new Hitable*[2];
        lightList[0] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(343.0, 548.5, 332.0),Vec3(213.0, 548.5, 332.0), light);
        lightList[1] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(213.0, 548.5, 332.0),Vec3(213.0, 548.5, 227.0), light);



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
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2), mirror);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), mirror);

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

        // //camera wall (we can see through this due to the directionality of Polygon_Ts)
        // //uses white material
        // /*
        // 549.6   0.0 0 
        // 0.0   0.0 0
        // 0.0 548.8 0
        // 556.0 548.8 0
        // */
        // device_object_list[i++] = Quad(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), mirror);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0), mirror);


        // //project the mesh to the proper place and scale it
        for(int j = 0; j < mesh_lengths[1]; j++) {
            //rotate about y 180 degrees
            float x = meshes[1][j].x;
            float z = meshes[1][j].z;
            meshes[1][j].x = x * cos(M_PI) - z * sin(M_PI);
            meshes[1][j].z = x * sin(M_PI) + z * cos(M_PI);


            meshes[1][j].x = meshes[1][j].x * 75.0f + 278.0f;
            meshes[1][j].y = meshes[1][j].y * 75.0f + 75.0f; 
            meshes[1][j].z = meshes[1][j].z * 75.0f + 278.0f;
        }

        //matte marble material
        Material * marble = new LambertianBordered(Vec3(0.9, 0.9, 0.9));

        // for(int j = 0; j < mesh_lengths[1]/3; j++) {
        //     device_object_list[i++] = Triangle(meshes[1][j*3], meshes[1][j*3 + 1], meshes[1][j*3 + 2], marble);
        // }


        // Vec3 lookfrom(278.0f, 278.0f, -400.0f);
        Vec3 lookfrom(400.0f, 278.0f, -400.0f);


        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        // *d_world = new Scene(device_object_list, i);
        *d_world  = new Octree(device_object_list, i);
        // //initialize Octree
        // printf("Initializing Octree\n");
        ((Octree*)*d_world)->max_depth = 4;
        // printf("Max depth set\n");
        ((Octree*)*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);
        // printf("Octree initialized\n");
        (*d_world)->setLights(lightList, 2);

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
        (*d_camera)->samples = 100;
        (*d_camera)->bounces = 20;


        // printf("World created\n");
        // (*d_world)->debug_print();
    }
}

#endif