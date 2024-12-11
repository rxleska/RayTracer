#include "headers/ObjInst.hpp"


__device__ bool ObjInstTrans::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
    // translate the ray to the object's frame
    Ray moved = Ray(r.origin - translation, r.direction);

    if(!obj->hit(moved, t_min, t_max, rec, state)) {
        return false;
    }
    rec.p = rec.p + translation;

    return true;
}

__device__ void ObjInstTrans::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const {
    obj->getBounds(x_min, x_max, y_min, y_max, z_min, z_max);
    x_min += translation.x;
    x_max += translation.x;
    y_min += translation.y;
    y_max += translation.y;
    z_min += translation.z;
    z_max += translation.z;
}

__device__ bool ObjInstTrans::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const {
    return obj->insideBox(x_min - translation.x, x_max - translation.x, y_min - translation.y, y_max - translation.y, z_min - translation.z, z_max - translation.z);
}

#include <iostream>

__device__ void ObjInstTrans::debug_print() const {
    printf("ObjInstTrans: translation: ");
    printf("x: %f, y: %f, z: %f\n", translation.x, translation.y, translation.z);
    obj->debug_print();    
}

__device__ Vec3 ObjInstTrans::getRandomPointInHitable(curandState *state) const {
    return obj->getRandomPointInHitable(state) + translation;
}
__device__ float ObjInstTrans::get2dArea() const {
    return obj->get2dArea();
}

#ifndef F_EPSILON
#define F_EPSILON 1e-6
#endif


__device__ bool ObjInstRot::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
    Ray moved = Ray(r.origin, r.direction);
    
    if(rot.x < F_EPSILON && rot.y < F_EPSILON && rot.x > -F_EPSILON && rot.y > -F_EPSILON){
        //z axis rotation (this check can save a lot of time)
        //todo: invert this rotation
        moved.origin.x = cos(rot.z) * r.origin.x - sin(rot.z) * r.origin.y;
        moved.origin.y = sin(rot.z) * r.origin.x + cos(rot.z) * r.origin.y;
        moved.direction.x = cos(rot.z) * r.direction.x - sin(rot.z) * r.direction.y;
        moved.direction.y = sin(rot.z) * r.direction.x + cos(rot.z) * r.direction.y;
    }
    else{
        float cosx,cosy,cosz,sinx,siny,sinz;
        //todo: invert this rotation
        cosx = cos(rot.x);
        cosy = cos(rot.y);
        cosz = cos(rot.z);
        sinx = sin(rot.x);
        siny = sin(rot.y);
        sinz = sin(rot.z);

        moved.origin.x = cosz*cosy*r.origin.x + (cosz*siny*sinx - sinz*cosx)*r.origin.y + (cosz*siny*cosx + sinz*sinx)*r.origin.z;
        moved.origin.y = sinz*cosy*r.origin.x + (sinz*siny*sinx + cosz*cosx)*r.origin.y + (sinz*siny*cosx - cosz*sinx)*r.origin.z;
        moved.origin.z = -siny*r.origin.x + cosy*sinx*r.origin.y + cosy*cosx*r.origin.z;

        moved.direction.x = cosz*cosy*r.direction.x + (cosz*siny*sinx - sinz*cosx)*r.direction.y + (cosz*siny*cosx + sinz*sinx)*r.direction.z;
        moved.direction.y = sinz*cosy*r.direction.x + (sinz*siny*sinx + cosz*cosx)*r.direction.y + (sinz*siny*cosx - cosz*sinx)*r.direction.z;
        moved.direction.z = -siny*r.direction.x + cosy*sinx*r.direction.y + cosy*cosx*r.direction.z;

    }
}

__device__ void ObjInstRot::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const {
    obj->getBounds(x_min, x_max, y_min, y_max, z_min, z_max);
    //TODO: rotate the bounds
}

__device__ bool ObjInstRot::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const {
    //TODO: rotate the box
}

#include <iostream>

__device__ void ObjInstRot::debug_print() const {
    printf("ObjInstRot: translation: ");
    printf("x: %f, y: %f, z: %f\n", rot.x, rot.y, rot.z);
    obj->debug_print();    
}

__device__ Vec3 ObjInstRot::getRandomPointInHitable(curandState *state) const {
    // todo rotate the point
    return obj->getRandomPointInHitable(state);
}
__device__ float ObjInstRot::get2dArea() const {
    return obj->get2dArea();
}