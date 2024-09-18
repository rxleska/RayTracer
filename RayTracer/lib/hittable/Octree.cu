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

    subdivide(0);
}
__device__ void Octree::subdivide(int depth){

}
__device__ int  Octree::closest_child(const Vec3 point) const{

    return 0;
}