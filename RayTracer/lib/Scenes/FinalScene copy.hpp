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

        // if(true){
            Material *white = new Lambertian(Vec3(1.0, 1.0, 1.0));
            // Material *white = new LambertianBordered(Vec3(1.0, 1.0, 1.0));
            Material *light = new Light(Vec3(1.0, 1.0, 1.0), 10.0f);
            // Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15)*(1.0f/0.45f));
            // Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05)*(1.0f/0.65f));
            Material *green = new Lambertian(Vec3(0.12, 0.45, 0.15));
            // Material *green = new LambertianBordered(Vec3(0.12, 0.45, 0.15));
            Material *red = new Lambertian(Vec3(0.65, 0.05, 0.05));
            // Material *red = new LambertianBordered(Vec3(0.65, 0.05, 0.05));
            // Material * mirror = new Metal(Vec3(0.9, 0.9, 0.9), 0.001);
            // Material * text0 = new Textured(textures[0], 474, 266);

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

            
            lightList[j++] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(343.0, 548.5, 332.0),Vec3(213.0, 548.5, 332.0), light);
            lightList[j++] = Triangle(Vec3(343.0, 548.5, 227.0),Vec3(213.0, 548.5, 332.0),Vec3(213.0, 548.5, 227.0), light);


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
            // device_object_list[i++] = Quad(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), text0, Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0));
            // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2), mirror);
            // device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), mirror);
            device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 0.0, 559.2),Vec3(0.0, 548.8, 559.2), white);
            device_object_list[i++] = Triangle(Vec3(549.6, 0.0, 559.2),Vec3(0.0, 548.8, 559.2),Vec3(556.0, 548.8, 559.2), white);

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
            //device_object_list[i++] = Quad(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
            device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(290.0, 165.0, 114.0),Vec3(240.0, 165.0, 272.0), white);
            device_object_list[i++] = Triangle(Vec3(290.0, 0.0, 114.0),Vec3(240.0, 165.0, 272.0),Vec3(240.0, 0.0, 272.0), white);
            
            
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

        // }

        // Material * redb = new LambertianBordered(Vec3(0.9, 0.2, 0.1),Vec3(0.9, 0.1, 0.9));
        // Material * red = new Light(Vec3(1.0f,0.0f,0.0f), 15);
        // Material * redb = new Lambertian(Vec3(1.0f,0.2f,0.1f));

        // //rotate mesh about x axis by 90 degrees
        // for(int j = 0; j < mesh_lengths[0]; j++) {
        //     float y = meshes[0][j].y;
        //     float z = meshes[0][j].z;
        //     meshes[0][j].y = y * cos(-M_PI/2) - z * sin(-M_PI/2);
        //     meshes[0][j].z = y * sin(-M_PI/2) + z * cos(-M_PI/2);

        //     float x = meshes[0][j].x;
        //     z = meshes[0][j].z;
        //     meshes[0][j].x = x * cos(M_PI) - z * sin(M_PI);
        //     meshes[0][j].z = x * sin(M_PI) + z * cos(M_PI);


        //     meshes[0][j] = meshes[0][j] * 45.0f;
        //     meshes[0][j] = meshes[0][j] + Vec3(278.0f, 0.0f , 278.0f);
        // }

        // for(int j = 0; j < mesh_lengths[0]/3; j++) {
        //     device_object_list[i++] = Triangle(meshes[0][j*3], meshes[0][j*3 + 1], meshes[0][j*3 + 2], redb);
        // }

        //flip texture 0 vertically
        // for(int j = 0; j < 474*266/2; j++) {
        //     Vec3 temp = textures[0][j];
        //     textures[0][j] = textures[0][474*266 - j - 1];
        //     textures[0][474*266 - j - 1] = temp;
        // }


        // Material * text = new Textured(textures[0], 474, 266);



        // ((Textured*)text)->rot = 0.25f;
        // // device_object_list[i++] = new Sphere(Vec3(0, 7.8, 0), 0.8, text);
        // Hitable * monkeyFace = new Sphere(Vec3(0, 0, 0), 0.8, text);
        // // monkeyFace = new ObjInstRot(monkeyFace, Vec3(11.0*M_PI/6.0, 0.0, 0.0));
        // monkeyFace = new ObjInstTrans(monkeyFace, Vec3(0, 7.8, 0));
        // device_object_list[i++] = monkeyFace;




        // circle light 
        // Material * light_mat = new Light(Vec3(1.0f,1.0f,1.0f),15);

        // device_object_list[i++] = new Sphere(Vec3(0.0f, -30.0f, 40.0f), 30, light_mat);
        // // device_object_list[i++] = new Sphere(Vec3(0.0f, 30.0f, 40.0f), 30, light_mat);

        // Hitable **lightList = new Hitable*[5];
        // int k = 0;
        // lightList[k++] = new Sphere(Vec3(0.0f, -30.0f, 40.0f), 30, light_mat);
        // // lightList[k++] = new Sphere(Vec3(0.0f, 30.0f, 40.0f), 30, light_mat);


        Material * test_mat = new Lambertian(Vec3(82.0f/255.0f, 178.0f/255.0f, 191.0f/255.0f));
        // Material * test_mat = new Metal(Vec3(0.7, 0.6, 0.5), 0.5);
        // Material * test_mat = new Dielectric(5.0);
        // Material * test_mat = new Light(Vec3(0.71f,0.48f,1.0f), 15);
        // lightList[j++] = new Sphere(Vec3(276.4, 200, 279.6), 100, test_mat);
        // Material * test_mat = new Textured(textures[1], 474, 327);
        // device_object_list[i++] = new Sphere(Vec3(276.4, 200, 279.6), 100, test_mat);

        // Material * smoke = new Isotropic(Vec3(0.8, 0.8, 0.8));
        // device_object_list[i++] = new Medium(Vec3(270.0, 185.0, 94.0), Vec3(330.0, 245.0, 154.0), 0.01, smoke);
        // device_object_list[i++] = new Medium(Vec3(0.0, 0, 0), Vec3(552.8, 145.0, 552.8), 0.01, smoke);

        // device_object_list[i++] = Quad(Vec3(276.4, 100, 220.6), Vec3(176.4, 200, 279.6), Vec3(276.4, 200, 279.6), Vec3(276.4, 100, 220.6), test_mat);
        
        // device_object_list[i++] = Quad(Vec3(250,100,250), Vec3(250,100,450), Vec3(450,100,450), Vec3(450,100,250), test_mat); 
        // //inverse
        // device_object_list[i++] = Quad(Vec3(250,100,250), Vec3(450,100,250), Vec3(450,100,450), Vec3(250,100,450), test_mat);
        
        // Vec3 *pentagon = new Vec3[5];
        // pentagon[0] = Vec3(250-50,100,250-100); //bottom left
        // pentagon[1] = Vec3(250+50,100,250-100); //bottom right
        // pentagon[2] = Vec3(250+81,100,250); //top right
        // pentagon[3] = Vec3(250,100,250+50); //tip
        // pentagon[4] = Vec3(250-81,100,250); //top left

        // device_object_list[i++] = new Polygon_T(pentagon, 5, test_mat);

        // Vec3 *pentagonBW = new Vec3[5];
        // pentagonBW[0] = pentagon[4];
        // pentagonBW[1] = pentagon[3];
        // pentagonBW[2] = pentagon[2];
        // pentagonBW[3] = pentagon[1];
        // pentagonBW[4] = pentagon[0];

        // device_object_list[i++] = new Polygon_T(pentagonBW, 5, test_mat);

        // Hitable * test_hitable = new Box(Vec3(-50, -50, -50), Vec3(50, 50, 50), test_mat);
        // test_hitable = new ObjInstRot(test_hitable, M_PI/4.0);
        // test_hitable = new ObjInstTrans(test_hitable, Vec3(276.4, 200, 279.6));
        


        // Hitable * test_hitable = new Sphere(Vec3(276.4, 200, 279.6), 30, test_mat);
        // test_hitable = new ObjInstMotion(test_hitable, Vec3(15,20,0), Vec3(0,-9.8,0), 10);
        // // Hitable * test_hitable2 = new Sphere(Vec3(276.4, 200, 279.6), 30, test_mat);
        // // test_hitable2 = new ObjInstMotion(test_hitable2, Vec3(-10,20,0), Vec3(0,-9.8,0), 10);

        // device_object_list[i++] = test_hitable;
        // // device_object_list[i++] = test_hitable2;
        // device_object_list[i++] = new Box(Vec3(276.4-30, 200-50, 279.6-30),Vec3(276.4+30, 200-30, 279.6+30), white);










        // Vec3 lookfrom(10,20,20);
        // Vec3 lookfrom(150.0f, 350.0f, -400.0f);
        Vec3 lookfrom(278.0f, 278.0f, -400.0f);
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

        Vec3 lookat(278.0f, 278.0f, 0.0f);
        // Vec3 lookat(278.0f, 278.0f, 250.0f);
        // Vec3 lookat(0,3,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 65.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
        (*d_camera)->ambient_light_level = 0.0f;
        (*d_camera)->msaa_x = 1;
        (*d_camera)->samples = 200;
        (*d_camera)->bounces = 10;
    }
}

#endif