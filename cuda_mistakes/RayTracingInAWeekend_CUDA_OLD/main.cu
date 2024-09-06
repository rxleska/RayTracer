

#include "lib/headers/RTWeekend.hpp"
#include "lib/headers/tppm.hpp"
// #include <vector> // arraylist, using to store objects in the image
#include "lib/headers/ArrayList.hpp"
#include "lib/headers/hittable.hpp"
#include "lib/headers/hittable_list.hpp"
#include "lib/headers/sphere.hpp"
#include "lib/headers/camera.hpp"

#define WIDTH 400
#include "lib/headers/Global.hpp"

#include <iostream>

int main(int argc, char** argv){
    // define the world
    hittable_list world;


    lambertian * ground_material = new lambertian(color(0.5, 0.5, 0.5));
    world.add(new sphere(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    vec3 albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    vec3 albedo = color::random(0.5, 1);
                    double fuzz = random_double(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    world.add(new sphere(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    // auto material = make_shared<lambertian>(color(1,0,0));
    // world.add(make_shared<sphere>(point3(5, 0.3, 2), 0.3, material));

    dielectric * material1 = new dielectric(1.5);
    world.add(new sphere(point3(0, 1, 0), 1.0, material1));

    lambertian* material2 = new lambertian(color(0.4, 0.2, 0.1));
    world.add(new sphere(point3(-4, 1, 0), 1.0, material2));

    // auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    metal* material3 = new metal(color(1, 0.8431, 0.0), 0.0);
    world.add(new sphere(point3(4, 1, 0), 1.0, material3));

    light* material4 = new light(color(1,1,1));
    world.add(new sphere(point3(0,510,400), 500, material4));
    


    // define the camera
    // camera cam(aspect_ratio, WIDTH, 100, 50, 90, point3(-2,2,1), point3(0,0,-1), vec3(0,1,0));
    camera cam = camera();

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1024;
    cam.samples_per_pixel = 16;
    cam.max_depth         = 30;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.0;
    cam.focus_dist    = 1.0;

    cam.render_cuda(world);
    
    return 0;
}