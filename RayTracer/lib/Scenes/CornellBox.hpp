#ifndef CORNELL_BOX_HPP
#define CORNELL_BOX_HPP

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


__device__ void create_Cornell_Box_Octree(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        Material *white = new Lambertian(Vec3(1.0, 1.0, 1.0));
        Material *light = new Light(Vec3(1.0, 1.0, 1.0), 10.0f);
        // Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15)*(1.0f/0.45f));
        // Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05)*(1.0f/0.65f));
        Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15));
        Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05));
        Material *mirror = new Metal(Vec3(0.9, 0.9, 0.9), 0.001);

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


        // add single point light at center of ceiling
        // Vec3 *pointLights = new Vec3[1*1*2];
        // int plc = 0;
        // pointLights[plc++] = Vec3(278.0, 548.8, 278.0);
        // pointLights[plc++] = Vec3(1.0, 1.0, 1.0);



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
        // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2), mirror);
        // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), mirror);

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
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), white);
        device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0), white);
        // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(0.0, 548.8, 0.0),Vec3(0.0, 0.0, 0.0), mirror);
        // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 0.0),Vec3(556.0, 548.8, 0.0),Vec3(0.0, 548.8, 0.0), mirror);

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
        // device_object_list[i++] = Quad(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
        device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0), white);
        device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
        
        // Material * text = new Textured(textures[1], 474, 327);
        // device_object_list[i++] = Quad(Vec3(290.0, 0.0, 114.0), Vec3(290.0, 165.0, 114.0), Vec3(240.0, 165.0, 272.0), Vec3(240.0, 0.0, 272.0), text, Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0));
        
        
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


        //little box test
        // Material * glass = new Dielectric(1.1);
        // device_object_list[i++] = new Box(Vec3(270.0, 185.0, 94.0), Vec3(330.0, 245.0, 154.0), glass);
        // device_object_list[i++] = new Box(Vec3(270.0, 185.0, 94.0), Vec3(310.0, 225.0, 134.0), red);
        // Material * blue = new Lambertian(Vec3(0.1, 0.2, 1.0));
        // device_object_list[i++] = new Box(Vec3(270.0, 185.0, 94.0), Vec3(330.0, 245.0, 154.0), red);

        // Hitable *rotationTest = new Sphere(Vec3(278.0, 278.0, 278.0), 50.0, red);
        // Hitable *rotationTest = new Box(Vec3(-30.0,-30.0,-30.0), Vec3(30.0,30.0,30.0), red);
        // rotationTest = new ObjInstRot(rotationTest, Vec3(0.0, M_PI/6, 0));
        // rotationTest = new ObjInstTrans(rotationTest, Vec3(450, 215, 124));
        // rotationTest = new ObjInstMotion(rotationTest, Vec3(20.0, 0.0, 0.0), Vec3(0.0,-9.8,0.0), 4.0);
        // device_object_list[i++] = rotationTest;

        // Material * smoke = new Isotropic(Vec3(0.8, 0.8, 0.8));
        // device_object_list[i++] = new Medium(Vec3(270.0, 185.0, 94.0), Vec3(330.0, 245.0, 154.0), 0.01, smoke);
        // device_object_list[i++] = new Medium(Vec3(0.0, 0, 0), Vec3(552.8, 145.0, 552.8), 0.01, smoke);
        // Material * white2 = new Lambertian(Vec3(1.0, 1.0, 1.0));
        // device_object_list[i++] = new Box(Vec3(0.0, 185.0, 94.0), Vec3(552.8, 225.0, 134.0), white2);


        Vec3 lookfrom(278.0f, 278.0f, -400.0f);
        // Vec3 lookfrom(200.0f, 278.0f, -400.0f);
        // Vec3 lookfrom(330.0f, 350.0f, -400.0f);
        // Vec3 lookfrom(278.0f, -278.0f, -400.0f);


        printf("rand initing\n");
        *rand_state = local_rand_state;
        // *d_world  = new Octree(device_object_list, i);
        printf("rand inited\n");
        // *d_world  = new Octree(device_object_list, i);
        *d_world = new Scene(device_object_list, i);
        // //initialize Octree
        // printf("Initializing Octree\n");
        // ((Octree*)*d_world)->max_depth = 2;
        // printf("Max depth set\n");
        // ((Octree*)*d_world)->init(lookfrom.x, lookfrom.y, lookfrom.z);
        // printf("Octree initialized\n");

        // (*d_world)->setPointLights(pointLights, plc);
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
        (*d_camera)->msaa_x = 1;
        (*d_camera)->samples = 200;
        (*d_camera)->bounces = 10;


        // printf("World created\n");
        // (*d_world)->debug_print();
    }
}

#endif