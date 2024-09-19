#ifndef OCTREE_HPP
#define OCTREE_HPP

#include "Scene.hpp"

class Octree : public Scene {
    public:
        Octree **children; // 8 children
        float x_min, x_max, y_min, y_max, z_min, z_max; // Bounding box
        int max_depth;
        Octree *parent;
        bool is_leaf = false;
        Vec3 center;

        __device__ Octree();
        __device__ Octree(Hitable **hitables, int hitable_count) : Scene(hitables, hitable_count);
        __device__ void init();
        __device__ void subdivide(int depth);
        __device__ int closest_child(const Vec3 point) const;
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;


};


#endif // OCTREE_HPP