#include "headers/Phong.hpp"

__device__ int Phong::scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    return 3;
}

__device__ Vec3 Phong::getColor(const Ray ray_out, const Vec3 *pointLights, int point_light_count, Vec3 ambient, curandState * rand_state) const{
    Vec3 color = ambient*this->kConsts.z; //k_a * i_a

    for(int i = 0; i < point_light_count; i++){
        //TODO
        //kd (L_m dot N) i_{m,d}

        // ks (R_m dot V)^a i_{m,s}
    } 

    return color;
}