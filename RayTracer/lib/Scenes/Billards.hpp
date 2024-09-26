#ifndef BILLARDS_HPP
#define BILLARDS_HPP

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


__device__ void create_Billards_Scene(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes){    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int object_count = 0;
        curandState local_rand_state = *rand_state;

        Material * light = new Light(Vec3(1.0f, 1.0f, 1.0f), 5.0f);

        Material * felt = new Lambertian(Vec3(3.0f/255.0f, 59.0f/255.0f, 186.0f/255.0f)); //blue felt
        Material * glossy = new Dielectric(1.5f); //glass
        Material * glossyl = new Dielectric(1.0f); //glass
        Material * glossym = new Dielectric(0.9f); //glass
        Material * yellowBalls = new Lambertian(Vec3(1.0f, 1.0f, 0.0f)); //yellow
        Material * redBalls = new Lambertian(Vec3(1.0f, 0.0f, 0.0f)); //red


        //current conversion 1 unit = 1 centifoot

        //table
        device_object_list[object_count++] = Quad(Vec3(0.0f, 0.0f, 0.0f),Vec3(-400.0f, 0.0f, 0.0f),Vec3(-400.0f, 0.0f, 700.0f),Vec3(0.0f, 0.0f, 700.0f), felt);


        //light 
        device_object_list[object_count++] = Quad(Vec3(0.0f, 1000.0f, 0.0f),Vec3(0.0f, 1000.0f, 700.0f),Vec3(-400.0f, 1000.0f, 700.0f),Vec3(-400.0f, 1000.0f, 0.0f), light);


        //balls size glossy 50.0f/3.0f, inner 45.0f/3.0f  
        //tostart we will put one in the center    
        device_object_list[object_count++] = new Sphere(Vec3(-200.0f, 50.0f/3.0f, 350.0f), 50.0f/3.0f, glossy);
        device_object_list[object_count++] = new Sphere(Vec3(-200.0f, 50.0f/3.0f, 350.0f), 33.0f/3.0f, yellowBalls);


        
        device_object_list[object_count++] = new Sphere(Vec3(-250.0f, 50.0f/3.0f, 350.0f), 50.0f/3.0f, glossyl);
        device_object_list[object_count++] = new Sphere(Vec3(-250.0f, 50.0f/3.0f, 350.0f), 33.0f/3.0f, yellowBalls);

        
        device_object_list[object_count++] = new Sphere(Vec3(-150.0f, 50.0f/3.0f, 350.0f), 50.0f/3.0f, glossym);
        device_object_list[object_count++] = new Sphere(Vec3(-150.0f, 50.0f/3.0f, 350.0f), 33.0f/3.0f, yellowBalls);




        Vec3 lookfrom(-200.0f, 250, -200.0f);


        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        // *d_world  = new Octree(device_object_list, i);
        *d_world = new Scene(device_object_list, object_count);

        Vec3 lookat(-200.0f, 50.0f/6.0f, 350.0f);
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 20.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
        (*d_camera)->ambient_light_level = 0.0f;
        (*d_camera)->msaa_x = 4;
        (*d_camera)->samples = 5000;
        (*d_camera)->bounces = 50;

    }
}

#endif