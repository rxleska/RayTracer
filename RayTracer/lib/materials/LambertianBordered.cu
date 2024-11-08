#include "headers/LambertianBordered.hpp"

__device__ int LambertianBordered::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    Vec3 normal = rec.normal; // get the normal of the hit point

    // get a random unit vector
    Vec3 bounceMod = Vec3::random(-10,10,rand_state); // 10 is arbitrary, since we are normalizing it later
    bounceMod.make_unit();

    // get the new direction
    Vec3 target = normal + bounceMod;
    
    // degenerate case where the new direction is close to zero
    if (target.isZero()) {
        target = normal;
    }

    //create new ray
    scattered_out = Ray(rec.p, target);

    //set the attenuation (color modification)
    if(rec.edge_hit){
        attenuation = border_color;
    }
    else{
        attenuation = albedo;
    }
    return 1;
}