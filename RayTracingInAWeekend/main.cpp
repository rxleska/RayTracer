

#include "lib/headers/RTWeekend.hpp"
#include "lib/headers/tppm.hpp"
#include <vector> // arraylist, using to store objects in the image
#include "lib/headers/hittable.hpp"
#include "lib/headers/hittable_list.hpp"
#include "lib/headers/sphere.hpp"
#include "lib/headers/polygon.hpp"
#include "lib/headers/camera.hpp"

#define WIDTH 400

#include <iostream>

// color ray_color(const ray& r, const hittable& world){
//     hit_record rec;
//     if(world.hit(r, interval(0, infinity), rec)){
//         return 0.5*color(rec.normal + color(1,1,1));
//     }


//     vec3 unit_direction = unit_vector(r.direction());
//     auto a = 0.5*(unit_direction.y() + 1.0);
//     return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
// }

void CornellBox(hittable_list &world, camera &cam){

    //floor
    auto white = make_shared<lambertian>(color(1, 1, 1));
    /*
    552.8 0.0   0.0   
    0.0 0.0   0.0
    0.0 0.0 559.2
    549.6 0.0 559.2
    */
    world.add(make_shared<polygon>(point3(552.8, 0.0, 0.0), point3(0.0, 0.0, 0.0), point3(0.0, 0.0, 559.2), white));
    world.add(make_shared<polygon>(point3(552.8, 0.0, 0.0), point3(0.0, 0.0, 559.2), point3(549.6, 0.0, 559.2), white));

    //light
    // auto l_mat = make_shared<light>(color(1, 200.0/255.0, 90/255.0));
    auto l_mat = make_shared<light>(color(1, 1, 1));
    /*
    343.0 548.8 227.0 
    343.0 548.8 332.0
    213.0 548.8 332.0
    213.0 548.8 227.0
    */
    world.add(make_shared<polygon>(point3(343.0, 548.5, 227.0), point3(343.0, 548.5, 332.0), point3(213.0, 548.5, 332.0), l_mat));
    world.add(make_shared<polygon>(point3(343.0, 548.5, 227.0), point3(213.0, 548.5, 332.0), point3(213.0, 548.5, 227.0), l_mat));
    //test whole ceiling as light
    // world.add(make_shared<polygon>(point3(556.0, 548.8, 0.0), point3(556.0, 548.8, 559.2), point3(0.0, 548.8, 559.2), l_mat));
    // world.add(make_shared<polygon>(point3(556.0, 548.8, 0.0), point3(0.0, 548.8, 559.2), point3(0.0, 548.8, 0.0), l_mat));
    //bigger light
    // world.add(make_shared<polygon>(point3(343.0+75.0, 548.5, 227.0-75.0), point3(343.0+75.0, 548.5, 332.0+75.0), point3(213.0-75.0, 548.5, 332.0+75.0), l_mat));
    // world.add(make_shared<polygon>(point3(343.0+75.0, 548.5, 227.0-75.0), point3(213.0-75.0, 548.5, 332.0+75.0), point3(213.0-75.0, 548.5, 227.0-75.0), l_mat));

    //Ceiling
    //uses white material
    /*
    556.0 548.8 0.0   
    556.0 548.8 559.2
    0.0 548.8 559.2
    0.0 548.8   0.0
    */
    world.add(make_shared<polygon>(point3(556.0, 548.8, 0.0), point3(556.0, 548.8, 559.2), point3(0.0, 548.8, 559.2), white));
    world.add(make_shared<polygon>(point3(556.0, 548.8, 0.0), point3(0.0, 548.8, 559.2), point3(0.0, 548.8, 0.0),  white));

    //back wall
    //uses white material
    /*
    549.6   0.0 559.2 
    0.0   0.0 559.2
    0.0 548.8 559.2
    556.0 548.8 559.2
    */
    world.add(make_shared<polygon>(point3(549.6, 0.0, 559.2), point3(0.0, 0.0, 559.2), point3(0.0, 548.8, 559.2), white));
    world.add(make_shared<polygon>(point3(549.6, 0.0, 559.2), point3(0.0, 548.8, 559.2), point3(556.0, 548.8, 559.2), white));

    //right wall
    // auto green = make_shared<lambertian>(color(0.12, 0.45, 0.15));
    auto green = make_shared<lambertian>(color(0.12, 0.45, 0.15)*(1/0.45));
    /*
    0.0   0.0 559.2   
    0.0   0.0   0.0
    0.0 548.8   0.0
    0.0 548.8 559.2
    */
    world.add(make_shared<polygon>(point3(0.0, 0.0, 559.2), point3(0.0, 0.0, 0.0), point3(0.0, 548.8, 0.0), green));
    world.add(make_shared<polygon>(point3(0.0, 0.0, 559.2), point3(0.0, 548.8, 0.0), point3(0.0, 548.8, 559.2), green));

    //left wall
    // auto red = make_shared<lambertian>(color(0.65, 0.05, 0.05));
    auto red = make_shared<lambertian>(color(0.65, 0.05, 0.05)*(1/0.65));
    /*
    552.8   0.0   0.0 
    549.6   0.0 559.2
    556.0 548.8 559.2
    556.0 548.8   0.0
    */
    world.add(make_shared<polygon>(point3(552.8, 0.0, 0.0), point3(549.6, 0.0, 559.2), point3(556.0, 548.8, 559.2), red));
    world.add(make_shared<polygon>(point3(552.8, 0.0, 0.0),  point3(556.0, 548.8, 559.2),  point3(556.0, 548.8, 0.0),red));

    // //camera wall (we can see through this due to the directionality of polygons)
    // //uses white material
    // /*
    // 549.6   0.0 0 
    // 0.0   0.0 0
    // 0.0 548.8 0
    // 556.0 548.8 0
    // */
    world.add(make_shared<polygon>(point3(549.6, 0.0, 0), point3(0.0, 548.8, 0), point3(0.0, 0.0, 0),  white));
    world.add(make_shared<polygon>(point3(549.6, 0.0, 0), point3(556.0, 548.8, 0), point3(0.0, 548.8, 0), white));



    //short block
    //uses white material
    //wall1
    /*
    130.0 165.0  65.0 
    82.0 165.0 225.0
    240.0 165.0 272.0
    290.0 165.0 114.0
    */
    world.add(make_shared<polygon>(point3(130.0, 165.0, 65.0), point3(82.0, 165.0, 225.0), point3(240.0, 165.0, 272.0), white));
    world.add(make_shared<polygon>(point3(130.0, 165.0, 65.0), point3(240.0, 165.0, 272.0), point3(290.0, 165.0, 114.0), white));
    //wall2
    /*
    290.0   0.0 114.0
    290.0 165.0 114.0
    240.0 165.0 272.0
    240.0   0.0 272.0
    */
    world.add(make_shared<polygon>(point3(290.0, 0.0, 114.0), point3(290.0, 165.0, 114.0), point3(240.0, 165.0, 272.0), white));
    world.add(make_shared<polygon>(point3(290.0, 0.0, 114.0), point3(240.0, 165.0, 272.0), point3(240.0, 0.0, 272.0), white));
    //wall3
    /*
    130.0   0.0  65.0
    130.0 165.0  65.0
    290.0 165.0 114.0
    290.0   0.0 114.0
    */
    world.add(make_shared<polygon>(point3(130.0, 0.0, 65.0), point3(130.0, 165.0, 65.0), point3(290.0, 165.0, 114.0), white));
    world.add(make_shared<polygon>(point3(130.0, 0.0, 65.0), point3(290.0, 165.0, 114.0), point3(290.0, 0.0, 114.0), white));
    //wall4
    /*
    82.0   0.0 225.0
    82.0 165.0 225.0
    130.0 165.0  65.0
    130.0   0.0  65.0
    */
    world.add(make_shared<polygon>(point3(82.0, 0.0, 225.0), point3(82.0, 165.0, 225.0), point3(130.0, 165.0, 65.0), white));
    world.add(make_shared<polygon>(point3(82.0, 0.0, 225.0), point3(130.0, 165.0, 65.0), point3(130.0, 0.0, 65.0), white));
    //wall5
    /*
    240.0   0.0 272.0
    240.0 165.0 272.0
    82.0 165.0 225.0
    82.0   0.0 225.0
    */
    world.add(make_shared<polygon>(point3(240.0, 0.0, 272.0), point3(240.0, 165.0, 272.0), point3(82.0, 165.0, 225.0), white));
    world.add(make_shared<polygon>(point3(240.0, 0.0, 272.0), point3(82.0, 165.0, 225.0), point3(82.0, 0.0, 225.0), white));

    //tall block
    //uses white material
    //wall1
    /*
    423.0 330.0 247.0
    265.0 330.0 296.0
    314.0 330.0 456.0
    472.0 330.0 406.0
    */
    world.add(make_shared<polygon>(point3(423.0, 330.0, 247.0), point3(265.0, 330.0, 296.0), point3(314.0, 330.0, 456.0), white));
    world.add(make_shared<polygon>(point3(423.0, 330.0, 247.0), point3(314.0, 330.0, 456.0), point3(472.0, 330.0, 406.0), white));
    //wall2
    /*
    423.0   0.0 247.0
    423.0 330.0 247.0
    472.0 330.0 406.0
    472.0   0.0 406.0
    */
    world.add(make_shared<polygon>(point3(423.0, 0.0, 247.0), point3(423.0, 330.0, 247.0), point3(472.0, 330.0, 406.0), white));
    world.add(make_shared<polygon>(point3(423.0, 0.0, 247.0), point3(472.0, 330.0, 406.0), point3(472.0, 0.0, 406.0), white));
    //wall3
    /*
    472.0   0.0 406.0
    472.0 330.0 406.0
    314.0 330.0 456.0
    314.0   0.0 456.0
    */
    world.add(make_shared<polygon>(point3(472.0, 0.0, 406.0), point3(472.0, 330.0, 406.0), point3(314.0, 330.0, 456.0), white));
    world.add(make_shared<polygon>(point3(472.0, 0.0, 406.0), point3(314.0, 330.0, 456.0), point3(314.0, 0.0, 456.0), white));
    //wall4
    /*
    314.0   0.0 456.0
    314.0 330.0 456.0
    265.0 330.0 296.0
    265.0   0.0 296.0
    */
    world.add(make_shared<polygon>(point3(314.0, 0.0, 456.0), point3(314.0, 330.0, 456.0), point3(265.0, 330.0, 296.0), white));
    world.add(make_shared<polygon>(point3(314.0, 0.0, 456.0), point3(265.0, 330.0, 296.0), point3(265.0, 0.0, 296.0), white));
    //wall5
    /*
    265.0   0.0 296.0
    265.0 330.0 296.0
    423.0 330.0 247.0
    423.0   0.0 247.0
    */
    world.add(make_shared<polygon>(point3(265.0, 0.0, 296.0), point3(265.0, 330.0, 296.0), point3(423.0, 330.0, 247.0), white));
    world.add(make_shared<polygon>(point3(265.0, 0.0, 296.0), point3(423.0, 330.0, 247.0), point3(423.0, 0.0, 247.0), white));
    

    // cam.vfov     = 35;
    // cam.lookfrom = point3(278, 278, -800);
    // cam.lookat   = point3(278, 278, 0);
    // cam.focus_dist = 35; 
    // cam.defocus_angle = 0.0;

    cam.vfov     = 70;
    cam.lookfrom = point3(278, 278, -400);
    cam.lookat   = point3(278, 278, 0);
    cam.focus_dist = 35; 
    cam.defocus_angle = 0.0;
    cam.aspect_ratio      = 9.0 / 9.0;
}

void RTIAW(hittable_list &world, camera &cam){
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    // world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));
    //flat ground
    world.add(make_shared<polygon>(point3(-1000, 0, -1000),point3(-1000, 0, 1000),point3(1000, 0, -1000), ground_material));
    world.add(make_shared<polygon>(point3(-1000, 0, 1000), point3(1000, 0, 1000), point3(1000, 0, -1000), ground_material));


    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    // auto material = make_shared<lambertian>(color(1,0,0));
    // auto material = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    // world.add(make_shared<sphere>(point3(5, 0.3, 2), 0.3, material));
    // world.add(make_shared<polygon>(point3(-5, 0, -3), point3(5, 0, -3), point3(5, 5, -3), material));
    // world.add(make_shared<polygon>(point3(-5, 0, -3), point3(5, 5, -3), point3(-5, 5, -3), material));


    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    // auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    auto material3 = make_shared<metal>(color(1, 0.8431, 0.0), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    auto material4 = make_shared<light>(color(1,1,1));
    world.add(make_shared<sphere>(point3(0,510,400), 200, material4));

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.aspect_ratio      = 16.0 / 9.0;

}


int main(int argc, char** argv){
    // define the world
    hittable_list world = hittable_list();

    

    // define the camera
    // camera cam(aspect_ratio, WIDTH, 100, 50, 90, point3(-2,2,1), point3(0,0,-1), vec3(0,1,0));
    camera cam = camera();
    cam.aspect_ratio      = 16.0 / 9.0; //defualt aspect ratio (changed by the scene)

    // RTIAW(world, cam);
    CornellBox(world, cam);

    // 
    // cam.aspect_ratio      = 9.0 / 9.0;
    cam.image_width       = 750;
    cam.samples_per_pixel = 500;
    cam.max_depth         = 50;

    cam.thread_count = 20;

    cam.render_mt(world);
    
    return 0;
}