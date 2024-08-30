#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <cmath>

#define epsilon 0.001

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
        float colormod[3];
        bool is_light;

    Sphere(Vec3 position, float radius){
        this->position = position;
        this->radius = radius;
        this->is_light = false;
    }
    void set_colormod(float r, float g, float b){
        this->colormod[0] = r;
        this->colormod[1] = g;
        this->colormod[2] = b;
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
        float color[3];
        Ray * prev;
        Ray * next;
    
    Ray(Vec3 origin, Vec3 direction){ 
        this->origin = origin;
        this->direction = direction;
        prev = nullptr;
        next = nullptr;
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

    void set_color(float r, float g, float b){
        this->color[0] = r;
        this->color[1] = g;
        this->color[2] = b;
    }
};

Vec3* ray_sphere_intersect(Ray* ray, Sphere* sphere) {
    Vec3 oc = ray->origin - sphere->position;

    float a = ray->direction.dot(ray->direction);  // Assuming ray->direction is normalized, a should be 1
    float b = 2.0f * oc.dot(ray->direction);
    float c = oc.dot(oc) - sphere->radius * sphere->radius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0 + epsilon) {
        return nullptr;
    }

    float sqrtDiscriminant = sqrt(discriminant);
    float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
    float t2 = (-b + sqrtDiscriminant) / (2.0f * a);

    // Check the first valid intersection
    if (t1 > 0) {
        return new Vec3(ray->origin + ray->direction * t1);
    }

    // If t1 is negative, check t2
    if (t2 > 0) {
        return new Vec3(ray->origin + ray->direction * t2);
    }

    // Both intersections are behind the ray
    return nullptr;
}



Ray * get_bounce(Ray * ray, Vec3 intersection, Sphere s){
    Vec3 normal = intersection - s.position;
    normal = normal / sqrt(normal.length_squared());
    Vec3 rnormal = ray->direction / sqrt(ray->direction.length_squared()); // ensure the ray direction is normalized
    Vec3 reflection = rnormal - normal * 2.0f * rnormal.dot(normal);
    // reflection = reflection * -1.0f;

    Ray * bounce = new Ray(intersection, reflection);

    return bounce;
}


#define WIDTH 1024
#define HEIGHT 768
#define PI 3.14159265359
#define BOUNCE_LIMIT 10
#define SUBSAMPLES 2

int main(int argc, char **argv){
    Camera * c = new Camera(Vec3(0, 0, -5), Vec3(0, 0, -1), PI/2.0);

    std::vector<Sphere> spheres;
    //create light back sphere
    Sphere light(Vec3(0, 0, -50), 40);
    light.is_light = true;
    light.set_colormod(1, 1, 1);
    spheres.push_back(light);

    //create wall spheres
    Sphere wall1(Vec3(0, 100000, 0), 99900); //ceiling
    Sphere wall2(Vec3(0, -100000, 0), 99900); //floor
    Sphere wall3(Vec3(100000, 0, 0), 99900); //right wall
    Sphere wall4(Vec3(-100000, 0, 0), 99900); //left wall
    Sphere wall5(Vec3(0, 0, -100000), 99900); //back wall
    Sphere wall6(Vec3(0, 0, 100000), 99900); //front wall

    //set wall colors to 0.95, 0.95, 0.95
    wall1.set_colormod(0.9, 0.9, 0.9);
    wall2.set_colormod(0.9, 0.9, 0.9);
    wall3.set_colormod(0.9, 0.9, 0.9);
    wall4.set_colormod(0.9, 0.9, 0.9);
    wall5.set_colormod(0.9, 0.9, 0.9);
    wall6.set_colormod(0.9, 0.9, 0.9);

    spheres.push_back(wall1);
    spheres.push_back(wall2);
    spheres.push_back(wall3);
    spheres.push_back(wall4);
    spheres.push_back(wall5);
    spheres.push_back(wall6);
    
    // //create sphere
    // Sphere s1(Vec3(0, 20, -50), 20);
    // s1.set_colormod(0.9, 0.0, 0.0);
    // spheres.push_back(s1);

    
    // Sphere s2(Vec3(0, -20, -50), 0);
    // s2.set_colormod(0.0, 0.0, 0.9);
    // spheres.push_back(s2);

    FILE * f = fopen("image.ppm", "w");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

    uint8_t image[HEIGHT][WIDTH][3];

    int i, j, si, sj;
    //create rays
    for(i = 0; i < WIDTH; i++){
        for(j = 0; j < HEIGHT; j++){
            float pxcolor[3] = {0, 0, 0};

            for(si = 0; si < SUBSAMPLES; si++){
                for(sj = 0; sj < SUBSAMPLES; sj++){


                    float x = (2 * ((i * SUBSAMPLES) + si + 0.5) / (float)(WIDTH * SUBSAMPLES) - 1) * tan(c->fov / 2.0) * (WIDTH * SUBSAMPLES) / (float)(HEIGHT * SUBSAMPLES);
                    float y = (1 - 2 * ((j * SUBSAMPLES) + sj + 0.5) / (float)(HEIGHT * SUBSAMPLES)) * tan(c->fov / 2.0);

                    Vec3 direction = Vec3(x, y, -1);
                    direction = direction / sqrt(direction.length_squared());

                    Ray * r = new Ray(c->position, direction);
                    r->set_color(1, 1, 1);

                    int bounces = BOUNCE_LIMIT;
                    while(true){
                        Vec3 * intersection = nullptr;
                        Sphere * closest = nullptr;
                        float closest_dist = 1000000;

                        for(Sphere s : spheres){
                            Vec3 * inter = ray_sphere_intersect(r, &s);
                            if(inter != nullptr){
                                float dist = (*inter - r->origin).length_squared();
                                if(dist < closest_dist && dist > epsilon){
                                    closest_dist = dist;
                                    closest = &s;
                                    intersection = inter;
                                }
                            }
                        }
                        if(closest == nullptr){
                            break;
                        }
                        //create bounce ray
                        Ray * bounce = get_bounce(r, *intersection, *closest);
                        bounce->set_color(closest->colormod[0], closest->colormod[1], closest->colormod[2]);
                        r->set_next(bounce);
                        bounce->set_prev(r);
                        r = bounce;

                        if(closest->is_light){
                            if(bounces == BOUNCE_LIMIT){
                                printf("Light hit\n");
                            }
                            else{
                                printf("Light hit after %d bounces\n", BOUNCE_LIMIT - bounces);
                            }

                            break;
                        }

                        if(bounces-- == 0){
                            r->set_color(0, 0, 0);
                            break;
                        }
                    }

                    // get color from bounce ray
                    float cl[3] = {r->color[0], r->color[1], r->color[2]};
                    while(r->get_prev() != nullptr){
                        r = r->get_prev();
                        // printf("Color: %f, %f, %f\n", r->color[0], r->color[1], r->color[2]);
                        cl[0] = (cl[0] * r->color[0]);
                        cl[1] = (cl[1] * r->color[1]);
                        cl[2] = (cl[2] * r->color[2]);
                    }

                    pxcolor[0] += cl[0] *1.0/(float)(SUBSAMPLES*SUBSAMPLES);
                    pxcolor[1] += cl[1] *1.0/(float)(SUBSAMPLES*SUBSAMPLES);
                    pxcolor[2] += cl[2] *1.0/(float)(SUBSAMPLES*SUBSAMPLES);


                }
            }


            image[j][i][0] = pxcolor[0] * 255;
            image[j][i][1] = pxcolor[1] * 255;
            image[j][i][2] = pxcolor[2] * 255;

            // printf("bounces: %d\n", bounces);
        }
        // print percentage
        printf("\rRendering: %d%%", (i+1)*100/WIDTH);
    }

    fwrite(image, 1, WIDTH * HEIGHT * 3, f);
    fclose(f);


    return 0;
}