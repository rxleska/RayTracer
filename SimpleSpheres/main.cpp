#include <stdio.h>
#include <stdint.h>
#include <cmath>


class Vec3{
    public: 
        float x;
        float y;
        float z;

    Vec3(float x, float y, float z){
        this->x = x;
        this->y = y;
        this->z = z;
    }

    Vec3(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }

    Vec3 operator+(Vec3 v){
        return Vec3(this->x + v.x, this->y + v.y, this->z + v.z);
    }

    Vec3 operator-(Vec3 v){
        return Vec3(this->x - v.x, this->y - v.y, this->z - v.z);
    }

    Vec3 operator*(float scalar){
        return Vec3(this->x * scalar, this->y * scalar, this->z * scalar);
    }

    Vec3 operator/(float scalar){
        return Vec3(this->x / scalar, this->y / scalar, this->z / scalar);
    }

    float dot(Vec3 v){
        return this->x * v.x + this->y * v.y + this->z * v.z;
    }

    Vec3 cross(Vec3 v){
        return Vec3(this->y * v.z - this->z * v.y, this->z * v.x - this->x * v.z, this->x * v.y - this->y * v.x);
    }

    float length_squared(){
        return this->x * this->x + this->y * this->y + this->z * this->z;
    }

    //copy constructor
    Vec3 * copy(){
        return new Vec3(this->x, this->y, this->z);
    }

    void print(){
        printf("Vec3(%f, %f, %f)\n", this->x, this->y, this->z);
    }
};

class Sphere{
    public:
        Vec3 position;
        float radius;
        uint8_t color[3];
        bool is_light;

    Sphere(Vec3 position, float radius){
        this->position = position;
        this->radius = radius;
        this->is_light = false;
    }
    void set_color(uint8_t r, uint8_t g, uint8_t b){
        this->color[0] = r;
        this->color[1] = g;
        this->color[2] = b;
    }
};

class Camera{
    public:
        Vec3 position;
        Vec3 direction;
        float fov;

    Camera(Vec3 position, Vec3 direction, float fov){
        this->position = position;
        this->direction = direction;
        this->fov = fov;
    }
};

class Ray{
    public:
        Vec3 origin;
        Vec3 direction;
        uint8_t color[3];
        Ray * prev;
        Ray * next;
    
    Ray(Vec3 origin, Vec3 direction){ 
        this->origin = origin;
        this->direction = direction;
    }

    void set_prev(Ray * prev){
        this->prev = prev;
    }

    void set_next(Ray * next){
        this->next = next;
    }

    Ray * get_prev(){
        return this->prev;
    }

    Ray * get_next(){
        return this->next;
    }

    void set_color(uint8_t r, uint8_t g, uint8_t b){
        this->color[0] = r;
        this->color[1] = g;
        this->color[2] = b;
    }
};

Vec3 * ray_sphere_intersect(Ray * ray, Sphere * sphere){
    Vec3 oc = ray->origin - sphere->position;

    float a = ray->direction.dot(ray->direction);
    float b = 2.0f * oc.dot(ray->direction);
    float c = oc.dot(oc) - sphere->radius * sphere->radius;

    float discriminant = b * b - 4 * a * c;

    if(discriminant < 0){
        return nullptr;
    }

    float t = (-b - sqrt(discriminant)) / (2.0f * a);
    if(t < 0){
        t = (-b + sqrt(discriminant)) / (2.0f * a);
        if(t < 0){
            return nullptr;
        }
    }

    Vec3* intersection = (ray->origin + ray->direction * t).copy();

    return intersection;
}

Ray * get_bounce(Ray * ray, Vec3 intersection, Sphere s){
    // Vec3 normal = intersection - s.position;
    Vec3 normal = intersection - s.position;
    normal = normal / sqrt(normal.length_squared());
    Vec3 rnormal = ray->direction / sqrt(ray->direction.length_squared()); // ensure the ray direction is normalized
    Vec3 reflection = rnormal - normal * 2.0f * rnormal.dot(normal);
    // reflection = reflection * -1.0f;

    Ray * bounce = new Ray(intersection, reflection);

    return bounce;
}


#define WIDTH 800
#define HEIGHT 800
#define PI 3.14159265359

int main(int argc, char **argv){
    Camera * c = new Camera(Vec3(0, 0, 0), Vec3(0, 0, -1), PI/6);

    Sphere * sun = new Sphere(Vec3(0, 0, 10001), 10000);
    sun->set_color(255, 255, 255);
    sun->is_light = true;

    Sphere * red = new Sphere(Vec3(0, 0, -20), 3);
    red->set_color(255, 0, 0);

    FILE * f = fopen("image.ppm", "w");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

    FILE * f2 = fopen("bouncedata.txt", "w");

    uint8_t image[HEIGHT][WIDTH][3];

    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            float x, y;
            float im = abs((float)i - ((float)WIDTH / 2.0));
            float jm = abs((float)j - ((float)HEIGHT / 2.0));

            if(i < WIDTH / 2){
                float theta = c->fov * (im / (WIDTH / 2.0));
                x = -tan(theta);
            }
            else{
                float theta = c->fov * (im / (WIDTH / 2.0));
                x = tan(theta);
            }

            if(j < HEIGHT / 2){
                float theta = c->fov * (jm / (HEIGHT / 2.0));
                y = tan(theta);
            }
            else{
                float theta = c->fov * (jm / (HEIGHT / 2.0));
                y = -tan(theta);
            }
            

            //print every 10 pixels
            // if(i%10 == 0 && j%10 == 0){
            //     printf("x: %f, y: %f\n", x, y);
            // }

            Vec3 direction = Vec3(x, y, -1);
            direction = direction / sqrt(direction.length_squared());
            Ray * r = new Ray(c->position, direction);
            r->color[0] = 255;
            r->color[1] = 255;
            r->color[2] = 255;
            do{ 
                Sphere * spheres[2] = {sun, red};
                Vec3 * intersections[2] = {ray_sphere_intersect(r, sun), ray_sphere_intersect(r, red)};
                // find the closest intersection
                Vec3 * closest = nullptr;
                float closest_distance = 1000000;
                Sphere * closest_sphere = nullptr;
                for(int i = 0; i < 2; i++){
                    // get the distance from the ray origin to the intersection
                    if(intersections[i] != nullptr){
                        float distance = sqrt((*intersections[i] - r->origin).length_squared());
                        if(distance < closest_distance){
                            fprintf(f2, "distance: %f\n", distance);
                            closest_distance = distance;
                            closest = intersections[i];
                            closest_sphere = spheres[i];
                        }
                    }                
                }
                if(closest_sphere != nullptr){
                    Ray * bounce = get_bounce(r, *closest, *closest_sphere);
                    bounce->set_color(closest_sphere->color[0], closest_sphere->color[1], closest_sphere->color[2]);
                    r->set_next(bounce);
                    bounce->set_prev(r);
                    r = bounce;
                    fprintf(f2, "c:[%f,%f,%f] b:[%f,%f,%f] red: %d, green: %d\n", closest->x, closest->y, closest->z, bounce->direction.x, bounce->direction.y, bounce->direction.z, r->color[0], r->color[1]);
                    if(closest_sphere->is_light){
                        printf("light hit\n");
                        break;
                    }
                }
                else{
                    r->set_color(0, 0, 0);
                    break;
                }

            } while(r->get_next() != nullptr);

            // trace color back to the origin
            uint8_t *color = r->color;
            while(r->get_prev() != nullptr){
                r = r->get_prev();
                // printf("color: %d, %d, %d\n", r->color[0], r->color[1], r->color[2]);
                // color bit mask with the current color
                color[0] = (color[0] & r->color[0]);
                color[1] = (color[1] & r->color[1]);
                color[2] = (color[2] & r->color[2]);
            }



            // test do not bounce yet
            image[j][i][0] = color[0];
            image[j][i][1] = color[1];
            image[j][i][2] = color[2];
        }
    }

    fwrite(image, 1, WIDTH * HEIGHT * 3, f);



    fclose(f);
    fclose(f2);


    return 0;
}