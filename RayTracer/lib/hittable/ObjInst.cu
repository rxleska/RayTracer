#include "headers/ObjInst.hpp"

// Translation

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


// Rotation

#ifndef F_EPSILON
#define F_EPSILON 1e-6
#endif

__device__ bool ObjInstRot::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
    Ray moved = Ray(r.origin, r.direction);
    
    if(rot.x < F_EPSILON && rot.y < F_EPSILON && rot.x > -F_EPSILON && rot.y > -F_EPSILON){
        //z axis rotation (this check can save a lot of time)
        //todo: invert this rotation
        float rotinv = -rot.z;
        float cosz = cos(rotinv);
        float sinz = sin(rotinv);

        moved.origin.x = cosz * r.origin.x - sinz * r.origin.y;
        moved.origin.y = sinz * r.origin.x + cosz * r.origin.y;
        moved.direction.x = cosz * r.direction.x - sinz * r.direction.y;
        moved.direction.y = sinz * r.direction.x + cosz * r.direction.y;

        if(!obj->hit(moved, t_min, t_max, rec, state)) {
            return false;
        }

        cosz = cos(rot.z);
        sinz = sin(rot.z);

        rec.p.x = cosz * rec.p.x - sinz * rec.p.y;
        rec.p.y = sinz * rec.p.x + cosz * rec.p.y;
        rec.normal.x = cosz * rec.normal.x - sinz * rec.normal.y;
        rec.normal.y = sinz * rec.normal.x + cosz * rec.normal.y;
        rec.normal.make_unit();
        return true;

    }
    else{
        float cosx,cosy,cosz,sinx,siny,sinz;
        //todo: invert this rotation
        cosx = cos(-rot.x);
        cosy = cos(-rot.y);
        cosz = cos(-rot.z);
        sinx = sin(-rot.x);
        siny = sin(-rot.y);
        sinz = sin(-rot.z);

        moved.origin.x = cosz*cosy*r.origin.x + (cosz*siny*sinx - sinz*cosx)*r.origin.y + (cosz*siny*cosx + sinz*sinx)*r.origin.z;
        moved.origin.y = sinz*cosy*r.origin.x + (sinz*siny*sinx + cosz*cosx)*r.origin.y + (sinz*siny*cosx - cosz*sinx)*r.origin.z;
        moved.origin.z = -siny*r.origin.x + cosy*sinx*r.origin.y + cosy*cosx*r.origin.z;

        moved.direction.x = cosz*cosy*r.direction.x + (cosz*siny*sinx - sinz*cosx)*r.direction.y + (cosz*siny*cosx + sinz*sinx)*r.direction.z;
        moved.direction.y = sinz*cosy*r.direction.x + (sinz*siny*sinx + cosz*cosx)*r.direction.y + (sinz*siny*cosx - cosz*sinx)*r.direction.z;
        moved.direction.z = -siny*r.direction.x + cosy*sinx*r.direction.y + cosy*cosx*r.direction.z;

        if(!obj->hit(moved, t_min, t_max, rec, state)) {
            return false;
        }

        cosx = cos(rot.x);
        cosy = cos(rot.y);
        cosz = cos(rot.z);
        sinx = sin(rot.x);
        siny = sin(rot.y);
        sinz = sin(rot.z);

        rec.p.x = cosz*cosy*rec.p.x + (cosz*siny*sinx - sinz*cosx)*rec.p.y + (cosz*siny*cosx + sinz*sinx)*rec.p.z;
        rec.p.y = sinz*cosy*rec.p.x + (sinz*siny*sinx + cosz*cosx)*rec.p.y + (sinz*siny*cosx - cosz*sinx)*rec.p.z;
        rec.p.z = -siny*rec.p.x + cosy*sinx*rec.p.y + cosy*cosx*rec.p.z;

        Vec3 norm;
        norm.x = cosz*cosy*rec.normal.x + (cosz*siny*sinx - sinz*cosx)*rec.normal.y + (cosz*siny*cosx + sinz*sinx)*rec.normal.z;
        norm.y = sinz*cosy*rec.normal.x + (sinz*siny*sinx + cosz*cosx)*rec.normal.y + (sinz*siny*cosx - cosz*sinx)*rec.normal.z;
        norm.z = -siny*rec.normal.x + cosy*sinx*rec.normal.y + cosy*cosx*rec.normal.z;
        rec.normal = norm.normalized();
        return true;

    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ void ObjInstRot::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const {
    obj->getBounds(x_min, x_max, y_min, y_max, z_min, z_max);
    if(rot.x < F_EPSILON && rot.y < F_EPSILON && rot.x > -F_EPSILON && rot.y > -F_EPSILON){
        //z axis rotation (this check can save a lot of time)
        float theta = rot.z;;
        while(theta > 2 * M_PI) theta -= 2 * M_PI;

        Vec3 p[4]; // 4 points of the box (0,1 mins, 2,3 maxs)
        
        //top down look
        if(theta < M_PI){
            if(theta < M_PI / 2){
                //bottom face
                p[0] = Vec3(x_min, y_min, z_min);
                p[1] = Vec3(x_max, y_min, z_min);
                // top face
                p[2] = Vec3(x_min, y_max, z_max);
                p[3] = Vec3(x_max, y_max, z_max);
            }
            else {
                //right face 
                p[0] = Vec3(x_max, y_min, z_min);
                p[1] = Vec3(x_max, y_min, z_min);
                // left face
                p[2] = Vec3(x_min, y_max, z_max);
                p[3] = Vec3(x_min, y_min, z_max);
            }
        }
        else{
            if(theta < 3 * M_PI / 2){
                // top face 
                p[0] = Vec3(x_min, y_max, z_min);
                p[1] = Vec3(x_max, y_max, z_min);
                // bottom face
                p[2] = Vec3(x_min, y_min, z_max);
                p[3] = Vec3(x_max, y_min, z_max);
            }
            else {
                // left face 
                p[0] = Vec3(x_min, y_max, z_min);
                p[1] = Vec3(x_min, y_max, z_min);
                // right face
                p[2] = Vec3(x_max, y_min, z_max);
                p[3] = Vec3(x_max, y_max, z_max);
            }
        }


        float x;
        float y;
        for(int i = 0; i < 4; i++){
            x = p[i].x;
            y = p[i].y;
            p[i].x = cos(theta) * x - sin(theta) * y;
            p[i].y = sin(theta) * x + cos(theta) * y;
        }

        x_min = min(p[0].x, p[1].x);
        y_min = min(p[0].y, p[1].y);
        z_min = min(p[0].z, p[1].z);
        x_max = max(p[2].x, p[3].x);
        y_max = max(p[2].y, p[3].y);
        z_max = max(p[2].z, p[3].z);

        return;
    }
    else{
        float cosx,cosy,cosz,sinx,siny,sinz;
        cosx = cos(rot.x);
        cosy = cos(rot.y);
        cosz = cos(rot.z);
        sinx = sin(rot.x);
        siny = sin(rot.y);
        sinz = sin(rot.z);
        float x,y,z;



        for(int i = 0; i < 8; i++){
            Vec3 inter = Vec3(x_min, y_min, z_min);
            if(i & 1) inter.x = x_max;
            if(i & 2) inter.y = y_max;
            if(i & 4) inter.z = z_max;   
            // rotation 
            x = cosz*cosy*inter.x + (cosz*siny*sinx - sinz*cosx)*inter.y + (cosz*siny*cosx + sinz*sinx)*inter.z;
            y = sinz*cosy*inter.x + (sinz*siny*sinx + cosz*cosx)*inter.y + (sinz*siny*cosx - cosz*sinx)*inter.z;
            z = -siny*inter.x + cosy*sinx*inter.y + cosy*cosx*inter.z;

            if(x < x_min) x_min = x;
            if(x > x_max) x_max = x;
            if(y < y_min) y_min = y;
            if(y > y_max) y_max = y;
            if(z < z_min) z_min = z;
            if(z > z_max) z_max = z;
        }

    }

}

__device__ bool ObjInstRot::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const {
    float bounds[6];
    getBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
    return x_min <= bounds[3] && x_max >= bounds[0] && y_min <= bounds[4] && y_max >= bounds[1] && z_min <= bounds[5] && z_max >= bounds[2];
}

#include <iostream>

__device__ void ObjInstRot::debug_print() const {
    printf("ObjInstRot: translation: ");
    printf("x: %f, y: %f, z: %f\n", rot.x, rot.y, rot.z);
    obj->debug_print();    
}

__device__ Vec3 ObjInstRot::getRandomPointInHitable(curandState *state) const {
    Vec3 pnt = obj->getRandomPointInHitable(state);
    float cosx,cosy,cosz,sinx,siny,sinz;
    cosx = cos(rot.x);
    cosy = cos(rot.y);
    cosz = cos(rot.z);
    sinx = sin(rot.x);
    siny = sin(rot.y);
    sinz = sin(rot.z);
    Vec3 returned;
    returned.x = cosz*cosy*pnt.x + (cosz*siny*sinx - sinz*cosx)*pnt.y + (cosz*siny*cosx + sinz*sinx)*pnt.z;
    returned.y = sinz*cosy*pnt.x + (sinz*siny*sinx + cosz*cosx)*pnt.y + (sinz*siny*cosx - cosz*sinx)*pnt.z;
    returned.z = -siny*pnt.x + cosy*sinx*pnt.y + cosy*cosx*pnt.z;
    return returned;
}

__device__ float ObjInstRot::get2dArea() const {
    return obj->get2dArea();
}


// Motion Blur

__device__ bool ObjInstMotion::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const {
    float rng = curand_uniform(state) * t;

    Vec3 translation = vel * rng  + acc * rng * rng * 0.5;
    // translate the ray to the object's frame
    Ray moved = Ray(r.origin - translation, r.direction);

    if(!obj->hit(moved, t_min, t_max, rec, state)) {
        return false;
    }
    rec.p = rec.p + translation;

    return true;

}


__device__ void ObjInstMotion::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const {
    obj->getBounds(x_min, x_max, y_min, y_max, z_min, z_max);
    Vec3 Cache[2];
    Cache[0] = Vec3(x_min, y_min, z_min);
    Cache[1] = Vec3(x_max, y_max, z_max);


    // iterative approach
    int step = 100;
    for(int i = 0; i <= step; i++){
        float rng = t * (float(i)/step);
        Vec3 translation = vel * rng  + acc * rng * rng * 0.5;
        if(x_min < Cache[0].x + translation.x) x_min = Cache[0].x + translation.x;
        if(x_max > Cache[1].x + translation.x) x_max = Cache[1].x + translation.x;
        if(y_min < Cache[0].y + translation.y) y_min = Cache[0].y + translation.y;
        if(y_max > Cache[1].y + translation.y) y_max = Cache[1].y + translation.y;
        if(z_min < Cache[0].z + translation.z) z_min = Cache[0].z + translation.z;
        if(z_max > Cache[1].z + translation.z) z_max = Cache[1].z + translation.z;
    }
}

__device__ bool ObjInstMotion::insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const {
    float bounds[6];
    getBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
    return x_min <= bounds[3] && x_max >= bounds[0] && y_min <= bounds[4] && y_max >= bounds[1] && z_min <= bounds[5] && z_max >= bounds[2];
}

__device__ void ObjInstMotion::debug_print() const {
    printf("ObjInstMotion: translation: ");
    printf("t: %f, vel: %f, %f, %f, acc: %f, %f, %f\n", t, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
    obj->debug_print();    
}

__device__ Vec3 ObjInstMotion::getRandomPointInHitable(curandState *state) const {
    Vec3 pnt = obj->getRandomPointInHitable(state);
    float rng = curand_uniform(state) * t;
    Vec3 translation = vel * rng  + acc * rng * rng * 0.5;
    return pnt + translation;
}

__device__ float ObjInstMotion::get2dArea() const {
    return obj->get2dArea();
}