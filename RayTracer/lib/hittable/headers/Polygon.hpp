#ifndef POLYGON_HPP
#define POLYGON_HPP


#include "Hitable.hpp"

// class to represent a polygon object in the scene (inherits from Hittable)
class Polygon: public Hitable{
    public:
        Vec3 * vertices;
        int num_vertices;
        Material * mat;
        Vec3 normal; //stored for faster computation
        float area; //stored for faster computation

        __device__ Polygon () : vertices(nullptr), num_vertices(0) {}

        __device__ Polygon (Vec3 * vertices, int num_vertices, Material * mat);
        // function to check if a ray hits the sphere
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

        __device__ bool is_coplanar() const;

        __device__ void calculate_normal_and_area();
};

__device__ Polygon * Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material * mat);
__device__ Polygon * Quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Material * mat);

#endif