#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "../hittable/hittable.h"

struct bvh_node{
    union {
        int left_node_index;
        hittable left_leaf;
    };
    union {
        int right_node_index;
        hittable right_leaf;
    };
    bool isLeaf;
    aabb bbox;
};

__host__ __device__ bool hit(const bvh_node* node_array, int bvh_node_index, ray& r, float t_min, float t_max, hit_record& record, ray& scattered, curandState* curandState){
    bvh_node this_node = node_array[bvh_node_index];

    if(!this_node.bbox.hit(r, {t_min,t_max})) return false;

    bool hit_left, hit_right;
    if(this_node.isLeaf){
        hit_left = this_node.left_leaf.hit(r, t_min, t_max, record, scattered, curandState);
        hit_right = this_node.right_leaf.hit(r, t_min, t_max, record, scattered, curandState);
    }
    else{
        hit_left = hit(node_array, this_node.left_node_index, r, t_min, t_max, record, scattered, curandState);
        hit_right = hit(node_array, this_node.right_node_index, r, t_min, t_max, record, scattered, curandState);
    }
    return hit_left || hit_right;
    
}



#endif // BVH_H