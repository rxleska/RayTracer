#ifndef Polygon_T_HPP
#define Polygon_T_HPP


#include "Hitable.hpp"

// class to represent a Polygon_T object in the scene (inherits from Hittable)
class Polygon_T: public Hitable{
    public:
        Vec3 * vertices;
        int num_vertices;
        Material * mat;
        Vec3 normal; //stored for faster computation
        float area; //stored for faster computation
        Vec3 * uvmap; // u,v coordinates for texture mapping optional
        bool use_uvmap = false;

        __device__ Polygon_T () : vertices(nullptr), num_vertices(0) {}

        __device__ Polygon_T (Vec3 * vertices, int num_vertices, Material * mat);

        __device__ Polygon_T(Vec3 * vertices, int num_vertices, Material * mat, Vec3 * uvmap);
        // function to check if a ray hits the sphere
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

        // function to get the bounding box of the Polygon_T
        __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const override;

        __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const override;

        __device__ bool is_coplanar() const;

        __device__ void calculate_normal_and_area();

        __device__ virtual  void debug_print() const override;


        __device__ virtual Vec3 getRandomPointInHitable(curandState *state) const override;


};

__device__ Polygon_T * Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material * mat);
__device__ Polygon_T * Quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Material * mat);

//uvmaped versions

__device__ Polygon_T * Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material * mat, Vec3 uv1, Vec3 uv2, Vec3 uv3);
__device__ Polygon_T * Quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Material * mat, Vec3 uv1, Vec3 uv2, Vec3 uv3, Vec3 uv4);



#endif