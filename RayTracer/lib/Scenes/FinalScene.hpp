#ifndef FINAL_SCENE_HPP
#define FINAL_SCENE_HPP

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

__device__ void create_final_scene(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        Material * red = new LambertianBordered(Vec3(0.9, 0.2, 0.1),Vec3(0.1, 0.2, 0.9));
        // Material * red = new Light(Vec3(1.0f,0.0f,0.0f), 15);
        // Material * red = new Lambertian(Vec3(1.0f,0.2f,0.1f));

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


        // circle light 
        Material * light_mat = new Light(Vec3(1.0f,1.0f,1.0f),15);

        device_object_list[i++] = new Sphere(Vec3(0.0f, -30.0f, 30.0f), 10, light_mat);


        Vec3 lookfrom(5,10,10);

        // printf("rand initing\n");
        *rand_state = local_rand_state;
        *d_world  = new Octree(device_object_list, i);
        ((Octree*)*d_world)->max_depth = 4;
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
        (*d_camera)->ambient_light_level = 0.6f;
        (*d_camera)->msaa_x = 1;
        (*d_camera)->samples = 1000;
        (*d_camera)->bounces = 64;
    }
}

#endif