#include "headers/Octree.hpp" 


#ifndef F_MAX
#define F_MAX 3.402823466e+38F
#endif

__device__ Octree::Octree(){
    hitable_count = 0;
    hitable_capacity = 10;
    hitables = (Hitable**)malloc(sizeof(Hitable*) * hitable_capacity);

    children = (Octree**)malloc(sizeof(Octree*) * 8);
}


__device__ Octree::Octree(Hitable **hitables, int hitable_count) : Scene(hitables, hitable_count){
    children = (Octree**)malloc(sizeof(Octree*) * 8);
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
    if(depth > max_depth){
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

        // for each hitable in the octree
        for(int i = 0; i < hitable_count; i++){
            Hitable * h = hitables[i];
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

    //TODO could free memory here
    
    //call subdivide on each child
    for(int i = 0; i < 8; i++){
        children[i]->max_depth = max_depth;
        children[i]->parent = this;
        children[i]->subdivide(depth + 1);
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

        //flip x
        //TODO WORKING HERE
        if(children[indexClosest]->hit(ray, t_min, t_max, rec)){
            return true;
        }
    }

    return has_hit;
}