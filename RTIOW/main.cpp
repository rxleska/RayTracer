

#include "lib/headers/RTWeekend.hpp"
#include "lib/headers/tppm.hpp"
#include <vector> // arraylist, using to store objects in the image
#include "lib/headers/hittable.hpp"
#include "lib/headers/hittable_list.hpp"
#include "lib/headers/sphere.hpp"
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

int main(int argc, char** argv){
    // calculate the aspect ratio
    auto aspect_ratio = 16.0 / 9.0;

    // define the world
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    // for (int a = -11; a < 11; a++) {
    //     for (int b = -11; b < 11; b++) {
    //         auto choose_mat = random_double();
    //         point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

    //         if ((center - point3(4, 0.2, 0)).length() > 0.9) {
    //             shared_ptr<material> sphere_material;

    //             if (choose_mat < 0.8) {
    //                 // diffuse
    //                 auto albedo = color::random() * color::random();
    //                 sphere_material = make_shared<lambertian>(albedo);
    //                 world.add(make_shared<sphere>(center, 0.2, sphere_material));
    //             } else if (choose_mat < 0.95) {
    //                 // metal
    //                 auto albedo = color::random(0.5, 1);
    //                 auto fuzz = random_double(0, 0.5);
    //                 sphere_material = make_shared<metal>(albedo, fuzz);
    //                 world.add(make_shared<sphere>(center, 0.2, sphere_material));
    //             } else {
    //                 // glass
    //                 sphere_material = make_shared<dielectric>(1.5);
    //                 world.add(make_shared<sphere>(center, 0.2, sphere_material));
    //             }
    //         }
    //     }
    // }

    auto material = make_shared<lambertian>(color(1,0,0));
    world.add(make_shared<sphere>(point3(5, 0.3, 2), 0.3, material));

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    // auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    auto material3 = make_shared<metal>(color(1, 0.8431, 0.0), 0.05);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    auto material4 = make_shared<light>(color(1,1,1));
    world.add(make_shared<sphere>(point3(0,510,400), 500, material4));
    


    // define the camera
    // camera cam(aspect_ratio, WIDTH, 100, 50, 90, point3(-2,2,1), point3(0,0,-1), vec3(0,1,0));
    camera cam = camera();

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1920;
    cam.samples_per_pixel = 20;
    cam.max_depth         = 10;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.0;
    cam.focus_dist    = 1.0;

    cam.render_mt(world);
    
    return 0;
}