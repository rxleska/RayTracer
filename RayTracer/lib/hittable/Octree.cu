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


__device__ void Octree::init(){
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

    center.x = (x_min + x_max) / 2;
    center.y = (y_min + y_max) / 2;
    center.z = (z_min + z_max) / 2;

    subdivide(0);
}
__device__ void Octree::subdivide(int depth){
    printf("Subdividing at depth %d\n", depth);
    if(depth > max_depth){
        is_leaf = true;
        return;
    }


    if(hitable_count <= 1){
        is_leaf = true;
        return;
    }

    // for each child node of the octree
    float xlen = x_max - x_min;
    float ylen = y_max - y_min;
    float zlen = z_max - z_min;
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
            children[i]->x_min = x_min + xlen / 2;
            children[i]->x_max = x_max;
        }
        else{ //xi = 0 use lower
            children[i]->x_min = x_min;
            children[i]->x_max = x_min + xlen / 2;
        }
        if(yi){//yi = 1 use higher
            children[i]->y_min = y_min + ylen / 2;
            children[i]->y_max = y_max;
        }
        else{//yi = 0 use lower
            children[i]->y_min = y_min;
            children[i]->y_max = y_min + ylen / 2;
        }
        if(zi){//zi = 1 use higher
            children[i]->z_min = z_min + zlen / 2;
            children[i]->z_max = z_max;
        }
        else{//zi = 0 use lower
            children[i]->z_min = z_min;
            children[i]->z_max = z_min + zlen / 2;
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
        return false;
    }

    if(t > t_max){
        return false;
    }

    bool has_hit = false;

    if(is_leaf){
        //check all hitables
        for(int i = 0; i < hitable_count; i++){
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
            return true;
        }
        //find next closest child
        //ideaology find the closest plane (based around center) to the ray in the direction of the ray
        Ray newRay = ray;
        float t_max = 0;
        for(int i = 0; i < 3; i++){ //a ray at max can pass through 4/8 children of a rectangular prism
            int plane = closest_plane(newRay, t_max);
            if(plane == 3){
                break;
            }
            indexClosest ^= 1 << plane; //flip the bit of the plane that was hit
            if(children[indexClosest]->hit(ray, t_min, t_max, rec)){
                return true;
            }
            //update the ray to the next closest plane
            newRay.origin = ray.origin + ray.direction * t_max + ray.direction * 0.0001f;
        }

    }

    return has_hit;
}


__device__ int Octree::closest_plane(const Ray &ray, float t) const{
        //return 0,1,2 or 3 (0:x) (1:y) (2:z) (none:3)

        //find t to each plane
        float t_x_min = (center.x - ray.origin.x) / ray.direction.x;
        float t_x_max = (center.x - ray.origin.x) / ray.direction.x;
        float t_y_min = (center.y - ray.origin.y) / ray.direction.y;

        //if any are negative set to max
        if(t_x_min < 0){
            t_x_min = F_MAX;
        }
        if(t_x_max < 0){
            t_x_max = F_MAX;
        }
        if(t_y_min < 0){
            t_y_min = F_MAX;
        }

        // TODO check if there is a faster way to do this
        float t_min = fminf(t_x_min, fminf(t_x_max, t_y_min));
        if (t_min == F_MAX){
            return 3;
        }
        if(t_min == t_x_min){
            return 0;
        }
        if(t_min == t_x_max){
            return 1;
        }
        if(t_min == t_y_min){
            return 2;
        }
        return 3;
}