#include "headers/Lambertian.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ int Lambertian::scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    Vec3 normal = rec.normal; // get the normal of the hit point

    // // get a random unit vector
    // Vec3 bounceMod = Vec3::random(-10,10,rand_state); // 10 is arbitrary, since we are normalizing it later
    // bounceMod.make_unit();

    // // get the new direction
    // Vec3 target = normal + bounceMod;
    
    Vec3 target = Vec3::random_on_hemisphere(rand_state, normal);

    // degenerate case where the new direction is close to zero
    if (target.isZero()) {
        target = normal;
    }

    //create new ray
    scattered_out = Ray(rec.p, target);

    //set the attenuation (color modification)
    attenuation = albedo;
    return 1;
}
__device__ double Lambertian::importance_pdf(const Ray &ray_in, const HitRecord &rec, const Ray &scattered, Vec3 *lightPoints, int lightCount) const {


    Vec3 dirToLight = Vec3(0,0,0);
    for(int i = 0; i < lightCount; i+=2){
        dirToLight = dirToLight + (lightPoints[i] - rec.p);
    }
    dirToLight.make_unit();


    // the pdf for lambertian is cos(theta) / pi
    // double cosine = rec.normal.dot(scattered.direction);
    double cosine = dirToLight.dot(scattered.direction);
    if (cosine < 0) {
        cosine = 0;
    }
    return cosine / M_PI;    
}
