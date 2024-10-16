#include "headers/Scene.hpp"

#include "headers/Hitable.hpp"
#include "../processing/headers/Ray.hpp"

#include "../materials/headers/Phong.hpp"
#include "../materials/headers/PhongLamb.hpp"

#include <iostream>


__device__ Scene::Scene(){
    hitable_count = 0;
    hitable_capacity = 2;
    // printf("Scene constructor\n");
    // hitables = new Hitable*[hitable_capacity]; // c++ new operator not supported in cuda
    hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_capacity);

    //point lights
    pointLights = nullptr;
    point_light_count = 0;
}

__host__ void Scene::free_memory(){
    cudaFree(hitables);
}

__device__ void Scene::resize(int new_capacity){
    // printf("Resizing\n");
    // Hitable **new_hitables = new Hitable*[new_capacity];
    Hitable **new_hitables = (Hitable**)malloc(sizeof(Hitable*) * new_capacity);
    for (int i = 0; i < hitable_count; i++) {
        new_hitables[i] = hitables[i];
    }
    // delete[] hitables;
    free(hitables);
    hitables = new_hitables;
    hitable_capacity = new_capacity;

}
__device__ void Scene::addHitable(Hitable *hittable){
    if (hitable_count == hitable_capacity) {
        resize(hitable_capacity * 2);
    }
    hitables[hitable_count++] = hittable;
}
__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const{
    bool has_hit = false;
    // float closest = t_max;
    for (int i = 0; i < hitable_count; i++) {
        if (hitables[i]->hit(ray, t_min, t_max, rec)) {
            has_hit = true;
            t_max = rec.t;
            // rec = current_hit;
        }
    }

    return has_hit;
}

__device__ Scene::Scene(Hitable **hitables, int hitable_count){
    this->hitable_count = hitable_count;
    this->hitable_capacity = hitable_count;
    this->hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_count);
    for (int i = 0; i < hitable_count; i++) {
        this->hitables[i] = hitables[i];
    }
}


__device__ void Scene::empty(){
    hitable_count = 0;
}


__device__ void Scene::debug_print() const{
    for(int i = 0; i < hitable_count; i++){
        hitables[i]->debug_print();
    }
}

__device__ void Scene::setPointLights(Vec3 *pointLights, int light_count){
    this->pointLights = (Vec3*)malloc(sizeof(Vec3) * light_count);
    for (int i = 0; i < light_count; i++) {
        this->pointLights[i] = pointLights[i];
    }
    this->point_light_count = light_count;
}


__device__ Vec3 Scene::handlePhong(const HitRecord &rec, Camera **cam) const{
    Phong *material = (Phong*) rec.mat;

    Vec3 returned_color = Vec3(1.0,1.0,1.0) * (*cam)->ambient_light_level * material->kConsts.z;

    // N_hat normal out of the surface
    Vec3 N_hat = rec.normal;
    N_hat.normalize();

    // vector towards the camera
    Vec3 V_hat = (*cam)->origin - rec.p;
    V_hat.normalize();

    for(int i = 0; i < point_light_count; i+=2){
        //vector towards the light
        Vec3 L_hat_m = pointLights[i] - rec.p; 
        L_hat_m.normalize();

        float Lm_dot_N = L_hat_m.dot(N_hat);

        float time = (pointLights[i] - rec.p).x / L_hat_m.x;
        Ray check_ray = Ray(rec.p, L_hat_m);
        HitRecord check_rec; //this is not used but a null pointer could lead to errors or undefined behavior
        if(!hit(check_ray, 0.001, time, check_rec)){
            //kd * Lm_dot_N * imd
            returned_color = returned_color + pointLights[i+1] * (Lm_dot_N * material->kConsts.y);

            Vec3 R_hat = (N_hat * 2.0f * Lm_dot_N ) - L_hat_m;
            R_hat.normalize();
            //ks * (R_hat dot V_hat)^a * ims
            float R_dot_V = R_hat.dot(V_hat);
            if(R_dot_V > 0){
                returned_color = returned_color + pointLights[i+1] * pow(R_dot_V, material->a) * material->kConsts.x;
            }
        }
        
    }


    return returned_color * material->albedo;
}

__device__ Vec3 Scene::handlePhongLamb(const HitRecord &rec, Camera **cam, Ray &scattered, curandState *local_rand_state, bool usePhong) const{
    PhongLamb *material = (PhongLamb*) rec.mat;

    if(usePhong){
        Vec3 returned_color = Vec3(1.0,1.0,1.0) * (*cam)->ambient_light_level * material->kConsts.z;

        // N_hat normal out of the surface
        Vec3 N_hat = rec.normal;
        N_hat.normalize();

        // vector towards the camera
        Vec3 V_hat = (*cam)->origin - rec.p;
        V_hat.normalize();

        for(int i = 0; i < point_light_count; i+=2){
            //vector towards the light
            Vec3 L_hat_m = pointLights[i] - rec.p; 
            L_hat_m.normalize();

            float Lm_dot_N = L_hat_m.dot(N_hat);

            float time = (pointLights[i] - rec.p).x / L_hat_m.x;
            Ray check_ray = Ray(rec.p, L_hat_m);
            HitRecord check_rec; //this is not used but a null pointer could lead to errors or undefined behavior
            if(!hit(check_ray, 0.001, time, check_rec)){
                //kd * Lm_dot_N * imd
                returned_color = returned_color + pointLights[i+1] * (Lm_dot_N * material->kConsts.y);

                Vec3 R_hat = (N_hat * 2.0f * Lm_dot_N ) - L_hat_m;
                R_hat.normalize();
                //ks * (R_hat dot V_hat)^a * ims
                float R_dot_V = R_hat.dot(V_hat);
                if(R_dot_V > 0){
                    returned_color = returned_color + pointLights[i+1] * pow(R_dot_V, material->a) * material->kConsts.x;
                }
            }
            
        }


        return returned_color * material->albedo;
    }
    else{
        Vec3 normal = rec.normal; // get the normal of the hit point

        // get a random unit vector
        Vec3 bounceMod = Vec3::random(-10,10,local_rand_state); // 10 is arbitrary, since we are normalizing it later
        bounceMod.make_unit();

        // get the new direction
        Vec3 target = normal + bounceMod;
        
        // degenerate case where the new direction is close to zero
        if (target.isZero()) {
            target = normal;
        }

        //create new ray
        scattered = Ray(rec.p, target);
        return material->albedo;
    }
}