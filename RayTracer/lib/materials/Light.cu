#include "headers/Light.hpp"

__device__ int Light::scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {

    // Light materials do not scatter rays they absorb them so we return 2, still set teh attenuation to the color of the light

    if(rec.edge_hit){
        attenuation = color * intensity * 0.1;
    }
    else{
        attenuation = color * intensity;
    }

    //TODO make light use the intensity modifier to make the light brighter or dimmer

    return 2;
}