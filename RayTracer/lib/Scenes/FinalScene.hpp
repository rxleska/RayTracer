#ifndef FINAL_SCENE_HPP
#define FINAL_SCENE_HPP

#include "../hittable/headers/Hitable.hpp"
#include "../hittable/headers/HitRecord.hpp"
#include "../hittable/headers/Octree.hpp"
#include "../hittable/headers/Sphere.hpp"
#include "../hittable/headers/Box.hpp"
#include "../hittable/headers/Medium.hpp"
#include "../hittable/headers/ObjInst.hpp"
#include "../hittable/headers/Polygon_T.hpp"
#include "../hittable/headers/Octree.hpp"
#include "../materials/headers/Material.hpp"
#include "../materials/headers/Lambertian.hpp"
#include "../materials/headers/Metal.hpp"
#include "../materials/headers/Dielectric.hpp"
#include "../materials/headers/Light.hpp"
#include "../materials/headers/Isotropic.hpp"
#include "../materials/headers/LambertianBordered.hpp"
#include "../materials/headers/Textured.hpp"
#include "../processing/headers/Camera.hpp"
#include "../processing/headers/Ray.hpp"
#include "../processing/headers/Vec3.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ void create_final_scene(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;
        //cornell box bounds are 0, 0, 0 to 552.8, 548.8, 559.2 (center at 276.4, 274.4, 279.6)
        Hitable **lightList = new Hitable*[20];
        int j = 0;

        Material * red        = new Metal(Vec3(0.65, 0.05, 0.05), 0.001);
        Material * green      = new Metal(Vec3(0.12, 0.45, 0.15), 0.001);
        Material * blue       = new Metal(Vec3(0.2, 0.4, 1.0), 0.005);
        Material * purple     = new Lambertian(Vec3(0.5, 0.1, 0.5));
        Material * white      = new Lambertian(Vec3(1.0, 1.0, 1.0));
        Material * light      = new Light(Vec3(1.0, 1.0, 1.0), 9.0f);
        Material * mirror     = new Metal(Vec3(0.9, 0.9, 0.9), 0.001);
        Material * glass      = new Dielectric(1.5);
        Material * blackSmoke = new Isotropic(Vec3(1.0, 0.1, 1.0));

        //floor
        device_object_list[i++] = Quad(
            Vec3(200.0,  0.0,  0.0),
            Vec3(  0.0,  0.0,  0.0),
            Vec3(  0.0,  0.0,200.0),
            Vec3(200.0,  0.0,200.0),
            white
        );

        //left wall
        device_object_list[i++] = Quad(
            Vec3(200.0,  0.0,  0.0),
            Vec3(200.0,  0.0,200.0),
            Vec3(200.0,200.0,200.0),
            Vec3(200.0,200.0,  0.0),
            red
        );

        //right wall
        device_object_list[i++] = Quad(
            Vec3(0.0,  0.0,  0.0),
            Vec3(0.0,200.0,  0.0),
            Vec3(0.0,200.0,200.0),
            Vec3(0.0,  0.0,200.0),
            green
        );

        //ceiling
        device_object_list[i++] = Quad(
            Vec3(200.0,200.0,  0.0),
            Vec3(200.0,200.0,200.0),
            Vec3(  0.0,200.0,200.0),
            Vec3(  0.0,200.0,  0.0),
            white
        );

        //back wall
        device_object_list[i++] = Quad(
            Vec3(200.0,  0.0,200.0),
            Vec3(  0.0,  0.0,200.0),
            Vec3(  0.0,200.0,200.0),
            Vec3(200.0,200.0,200.0),
            mirror
        );

        //front wall
        device_object_list[i++] = Quad(
            Vec3(200.0,  0.0,  0.0),
            Vec3(200.0,200.0,  0.0),
            Vec3(  0.0,200.0,  0.0),
            Vec3(  0.0,  0.0,  0.0),
            blue
        );
        



        

        device_object_list[i++] = new ObjInstTrans(
            new ObjInstRot(
                new Medium(
                    Vec3(-25, -25, -25),
                    Vec3(25, 25, 25),
                    0.03,
                    blackSmoke
                ), 
                Vec3(M_PI/4, 0.0, M_PI/3)
                ),
            Vec3(130,25,35)
        );

        //rotate mesh about x axis by 90 degrees
        for(int t = 0; t < mesh_lengths[0]; t++) {
            float y = meshes[0][t].y;
            float z = meshes[0][t].z;
            meshes[0][t].y = y * cos(-M_PI/2) - z * sin(-M_PI/2);
            meshes[0][t].z = y * sin(-M_PI/2) + z * cos(-M_PI/2);

            float x = meshes[0][t].x;
            z = meshes[0][t].z;
            meshes[0][t].x = x*cos(5.0f*M_PI/6.0f) - z*sin(5.0f*M_PI/6.0f);
            meshes[0][t].z = x*sin(5.0f*M_PI/6.0f) + z*cos(5.0f*M_PI/6.0f);

            meshes[0][t] = meshes[0][t] * 10.0f;
            meshes[0][t] = meshes[0][t] + Vec3(150.0f, 0.0f , 150.0f);
        }



        for(int t = 0; t < mesh_lengths[0]/3; t++) {
            device_object_list[i++] = Triangle(meshes[0][t*3], meshes[0][t*3 + 1], meshes[0][t*3 + 2], purple);
        }
        
        Material * text = new Textured(textures[1], 474, 327);


        device_object_list[i++] = new ObjInstTrans(
            new ObjInstRot(
                Quad(
                    Vec3(-35.0, 0.0, -35.0),
                    Vec3(-35.0, 0.0, 35.0),
                    Vec3(35.0, 0.0, 35.0),
                    Vec3(35.0, 0.0, -35.0),
                    text,
                    Vec3(0.0, 0.0, 0.0),
                    Vec3(0.0, 1.0, 0.0),
                    Vec3(1.0, 1.0, 0.0),
                    Vec3(1.0, 0.0, 0.0)
                ),
                Vec3(0.0, M_PI/4, 0.0)
            ),
            Vec3(50.0, 0.1, 100.0)
        );

         device_object_list[i++] = new ObjInstTrans(
            new ObjInstRot(
                Quad(
                    Vec3(-35.0, 0.0,-35.0),
                    Vec3(35.0, 0.0, -35.0),
                    Vec3(35.0, 0.0, 35.0),
                    Vec3(-35.0, 0.0,35.0),
                    text,
                    Vec3(0.0, 0.0, 0.0),
                    Vec3(1.0, 0.0, 0.0),
                    Vec3(1.0, 1.0, 0.0),
                    Vec3(0.0, 1.0, 0.0)
                ),
                Vec3(0.0, M_PI/4, 0.0)
            ),
            Vec3(50.0, 0.1, 100.0)
        );

        device_object_list[i++] = new Sphere(Vec3(50.0, 25.0, 100.0), 25.0, glass);



        float x,y,z,r, r_val, g_val, b_val, contrast_factor;
        contrast_factor = 0.6f;
        Material * random_mat;

        // random Spheres in a box (80,0,140) to (120,40,180)
        for(int t = 0; t < 100; t++) {
            x = 80.0 + 40.0*curand_uniform(&local_rand_state);
            y = 0.0 + 40.0*curand_uniform(&local_rand_state);
            z = 140.0 + 40.0*curand_uniform(&local_rand_state);
            r = 1.0 + 5.0*curand_uniform(&local_rand_state);

            r_val = curand_uniform(&local_rand_state)*0.5 + 0.5;
            g_val = curand_uniform(&local_rand_state)*0.5 + 0.5;
            b_val = curand_uniform(&local_rand_state)*0.5 + 0.5;
            r_val += curand_uniform(&local_rand_state) * contrast_factor - contrast_factor/2.0f;
            g_val += curand_uniform(&local_rand_state) * contrast_factor - contrast_factor/2.0f;
            b_val += curand_uniform(&local_rand_state) * contrast_factor - contrast_factor/2.0f;
            r_val = clamp(r_val, 0.0f, 1.0f);
            g_val = clamp(g_val, 0.0f, 1.0f);
            b_val = clamp(b_val, 0.0f, 1.0f);

            random_mat = new Lambertian(Vec3(r_val, g_val, b_val));
            device_object_list[i++] = new Sphere(Vec3(x, y, z), r, random_mat);    

        }

        
        Material* yellow_orangeB = new LambertianBordered(Vec3(0.8, 0.8, 0.2), Vec3(1.0, 0.447, 0.129));

        
        Vec3 pos = Vec3(40.0, 1.0, 40.0);
        float l = 30.0f;

        device_object_list[i++] = Triangle(
            Vec3(   0.0 + pos.x, 0.0 + pos.y,  l * sqrt(3.0)/4.0 +pos.z),
            Vec3(-l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            Vec3( l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            yellow_orangeB
        );
        device_object_list[i++] = Triangle(
            Vec3(   0.0 + pos.x, 0.0 + pos.y,  l * sqrt(3.0)/4.0 +pos.z),
            Vec3( 0.0 + pos.x, l * sqrt(3.0)/2.0+ pos.y, 0.0 +pos.z),
            Vec3(-l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            yellow_orangeB
        );
        device_object_list[i++] = Triangle(
            Vec3(   0.0 + pos.x, 0.0 + pos.y,  l * sqrt(3.0)/4.0 +pos.z),
            Vec3( l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            Vec3( 0.0 + pos.x, l * sqrt(3.0)/2.0+ pos.y, 0.0 +pos.z),
            yellow_orangeB
        );
        device_object_list[i++] = Triangle(
            Vec3(-l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            Vec3( 0.0 + pos.x, l * sqrt(3.0)/2.0+ pos.y, 0.0 +pos.z),
            Vec3( l/2.0 + pos.x, 0.0 + pos.y, -l * sqrt(3.0)/4.0 +pos.z),
            yellow_orangeB
        );



        //lights
        // Hitable * light1 = new Sphere(Vec3(200.0, 200.0, 100.0), 30.0, light);
        Hitable * light1 = Quad(
            Vec3(125.0,199.8, 75.0),
            Vec3(125.0,199.8,125.0),
            Vec3( 75.0,199.8,125.0),
            Vec3( 75.0,199.8, 75.0),
            light
        );
        // Hitable * light2 = Quad(
        //     Vec3(125.0,200.0, 75.0),
        //     Vec3( 75.0,200.0, 75.0),
        //     Vec3( 75.0,200.0,125.0),
        //     Vec3(125.0,200.0,125.0),
        //     light
        // );
        // Hitable * light3 = Quad(
        //     Vec3(325.0,200.0, 75.0),
        //     Vec3(325.0,200.0,125.0),
        //     Vec3(275.0,200.0,125.0),
        //     Vec3(275.0,200.0, 75.0),
        //     light
        // );
        // Hitable * light4 = Quad(
        //     Vec3(325.0,200.0, 75.0),
        //     Vec3(275.0,200.0, 75.0),
        //     Vec3(275.0,200.0,125.0),
        //     Vec3(325.0,200.0,125.0),
        //     light
        // );







        lightList[j++] = light1;
        device_object_list[i++] = light1;
        // lightList[j++] = light2;
        // device_object_list[i++] = light2;
        // lightList[j++] = light3;
        // device_object_list[i++] = light3;
        // lightList[j++] = light4;
        // device_object_list[i++] = light4;




        // Vec3 lookfrom(10,20,20);
        // Vec3 lookfrom(150.0f, 350.0f, -400.0f);
        Vec3 lookfrom(100, 125, -100.0f);
        // Vec3 lookfrom(278.0f, 450.0f, -400.0f);


        // printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        *d_world  = new Scene(device_object_list, i);
        // ((Octree*)*d_world)->max_depth = 4;
        // ((Octree*)*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);

        (*d_world)->setLights(lightList, j);

        // printf("rand inited\n");
        // *d_world  = new Scene(device_object_list, i);

        Vec3 lookat(100.0f, 100.0f, 50.0f);
        // Vec3 lookat(278.0f, 278.0f, 250.0f);
        // Vec3 lookat(0,3,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 75.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
        (*d_camera)->ambient_light_level = 0.0f;
        (*d_camera)->msaa_x = 1;
        (*d_camera)->samples = 100;
        (*d_camera)->bounces = 20;
    }
}

#endif