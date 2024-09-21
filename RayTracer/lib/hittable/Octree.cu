#include "headers/Octree.hpp" 

#include <iostream>

#ifndef F_MAX
#define F_MAX 3.402823466e+38F
#endif

__device__ Octree::Octree() : Scene(){
    children = (Octree**)malloc( 8 * sizeof(Octree*));
}


__device__ Octree::Octree(Hitable **hitables, int hitable_count) : Scene(hitables, hitable_count){
    children = (Octree**)malloc(8 * sizeof(Octree*));
}


__device__ void Octree::init(float camx, float camy, float camz){
    x_min = y_min = z_min = F_MAX;
    x_max = y_max = z_max = -F_MAX;

    for (int i = 0; i < hitable_count; i++) {
        float x_min_temp, x_max_temp, y_min_temp, y_max_temp, z_min_temp, z_max_temp;
        hitables[i]->getBounds(x_min_temp, x_max_temp, y_min_temp, y_max_temp, z_min_temp, z_max_temp);
        x_min = fminf(x_min, x_min_temp);
        x_max = fmaxf(x_max, x_max_temp);
        y_min = fminf(y_min, y_min_temp);
        y_max = fmaxf(y_max, y_max_temp);
        z_min = fminf(z_min, z_min_temp);
        z_max = fmaxf(z_max, z_max_temp);
    }

    x_min = fminf(x_min, camx)-10.0f;
    x_max = fmaxf(x_max, camx)+15.0f;
    y_min = fminf(y_min, camy)-10.0f;
    y_max = fmaxf(y_max, camy)+15.0f;
    z_min = fminf(z_min, camz)-10.0f;
    z_max = fmaxf(z_max, camz)+15.0f;



    center.x = (x_min + x_max) / 2;
    center.y = (y_min + y_max) / 2;
    center.z = (z_min + z_max) / 2;

    subdivide(0);
}
__device__ void Octree::subdivide(int depth){
    // printf("Subdividing at depth %d\n", depth);
    if(depth > max_depth){
        // printf("Max depth reached\n");
        is_leaf = true;
        return;
    }


    if(hitable_count <= 1){
        // printf("Max depth reached\n");
        is_leaf = true;
        return;
    }

    // for each child node of the octree
    // float xlen = x_max - x_min;
    // float ylen = y_max - y_min;
    // float zlen = z_max - z_min;
    int xi = 0;
    int yi = 0;
    int zi = 0;
    for(int i = 0; i < 8; i++){
        xi = i & 1;
        yi = (i >> 1) & 1;
        zi = (i >> 2) & 1;

        // create the child node
        children[i] = new Octree();
        if(xi){ //xi = 1 use higher
            children[i]->x_min = center.x;
            children[i]->x_max = x_max;
        }
        else{ //xi = 0 use lower
            children[i]->x_min = x_min;
            children[i]->x_max = center.x;
        }
        if(yi){//yi = 1 use higher
            children[i]->y_min = center.y;
            children[i]->y_max = y_max;
        }
        else{//yi = 0 use lower
            children[i]->y_min = y_min;
            children[i]->y_max = center.y;
        }
        if(zi){//zi = 1 use higher
            children[i]->z_min = center.z;
            children[i]->z_max = z_max;
        }
        else{//zi = 0 use lower
            children[i]->z_min = z_min;
            children[i]->z_max = center.z;
        }

        //create mid points
        children[i]->center.x = (children[i]->x_min + children[i]->x_max) / 2;
        children[i]->center.y = (children[i]->y_min + children[i]->y_max) / 2;
        children[i]->center.z = (children[i]->z_min + children[i]->z_max) / 2;


        // log child capactiy
        // printf("Child %d: x_min: %f, x_max: %f, y_min: %f, y_max: %f, z_min: %f, z_max: %f, cap: %d, cur %d\n", i, children[i]->x_min, children[i]->x_max, children[i]->y_min, children[i]->y_max, children[i]->z_min, children[i]->z_max, children[i]->hitable_capacity, children[i]->hitable_count);
        
        // for each hitable in the octree
        for(int j = 0; j < hitable_count; j++){
            // printf("Checking hitable %d\n", j);
            Hitable * h = hitables[j];
            if(h->insideBox(
                children[i]->x_min, children[i]->x_max,
                children[i]->y_min, children[i]->y_max,
                children[i]->z_min, children[i]->z_max
            )){
                // printf("adding hitable %d\n", j);
                children[i]->addHitable(h);
            }
        }
    }

    // remove all nodes from the parent
    empty();

    // so are we allocating this or not?
    free(hitables);
    
    //call subdivide on each child
    for(int k = 0; k < 8; k++){
        children[k]->max_depth = max_depth;
        // children[i]->parent = this;
        children[k]->subdivide(depth + 1);
    }
}
__device__ int  Octree::closest_child(const Vec3 point) const{
    Vec3 transposed = point - center;
    int xFlag = transposed.x > 0;
    int yFlag = transposed.y > 0;
    int zFlag = transposed.z > 0;

    return xFlag + 2 * yFlag + 4 * zFlag; // 0 to 7 index using same method as assignment
}

__device__ bool Octree::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const{
    float t;
    if(!ray.hitsBox(x_min, x_max, y_min, y_max, z_min, z_max, t)){
        // printf("No box hit\n");
        return false;
    }

    if(t > t_max){
        // printf("No box hit t_max\n");
        return false;
    }

    bool has_hit = false;

    if(is_leaf){
        //check all hitables
        // printf("Hit leaf\n");
        for(int i = 0; i < hitable_count; i++){
                // printf("Hit hittable\n");
            if(hitables[i]->hit(ray, t_min, t_max, rec)){
                has_hit = true;
                t_max = rec.t;
            }
        }
    }
    else{
        //determine the closest nodes to the ray origin
        int indexClosest = closest_child(ray.origin);

        if(children[indexClosest]->hit(ray, t_min, t_max, rec)){
            // printf("Hit child\n");
            return true;
        }
        //find next closest child
        //ideaology find the closest plane (based around center) to the ray in the direction of the ray

        float plane_ts[3];

        // x flip 
        plane_ts[0] = (center.x - ray.origin.x) / ray.direction.x;
        // y flip
        plane_ts[1] = (center.y - ray.origin.y) / ray.direction.y;
        // z flip
        plane_ts[2] = (center.z - ray.origin.z) / ray.direction.z;


        for(int iterator = 0; iterator < 3; iterator++){
            if(plane_ts[iterator] < 0.00001f){
                plane_ts[iterator] = F_MAX;
            }
        }

        for(int i = 0; i < 3; i++){ //a ray at max can pass through 4/8 children of a rectangular prism
            int plane = closest_plane(plane_ts);
            if(plane == 3){
                // printf("No plane hit\n");
                break;
            }
            // printf("Plane hit: %d\n", plane);
            indexClosest ^= 1 << plane; //flip the bit of the plane that was hit
            if(children[indexClosest]->hit(ray, t_min, t_max, rec)){
                // printf("Hit child\n");
                return true;
            }
            plane_ts[plane] = F_MAX;
        }

        //SLOW METHOD CHECKS ALL CHILDREN
        // for(int i = 0; i < 8; i++){
        //     if(children[i]->hit(ray, t_min, t_max, rec)){
        //         // printf("Hit child\n");
        //         return true;
        //     }
        // }


    }

    return has_hit;
}


__device__ int Octree::closest_plane(float *tvals) const {
    // Return 0 for x-plane, 1 for y-plane, 2 for z-plane, or 3 if none

    int plane = 3;
    float t = F_MAX;
    for(int i = 0; i < 3; i++){
        if(tvals[i] < t){
            t = tvals[i];
            plane = i;
        }
    }

    return plane;
}



__device__ void Octree::debug_print() const{
    printf("Octree: x_min: %f, x_max: %f, y_min: %f, y_max: %f, z_min: %f, z_max: %f\n", x_min, x_max, y_min, y_max, z_min, z_max);
    if(is_leaf){
        printf("Leaf:%d\n", hitable_count);
        // for(int i = 0; i < hitable_count; i++){
        //     hitables[i]->debug_print();
        // }
    }
    else{
        for(int i = 0; i < 8; i++){
            children[i]->debug_print();
        }
    }
    
}
