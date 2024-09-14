#ifndef HIT_RECORD_HPP
#define HIT_RECORD_HPP

#include "../../processing/headers/Vec3.hpp"
#include "../../materials/headers/Material.hpp"

// forward declaration of Material class
class Material;


// class to represent a hit record (when a ray hits an object)
class HitRecord{
    public:
        // time of hit (this is the time along a ray)
        float t;
        // point of hit
        Vec3 p;
        // normal of the hit object at the point of hit (used for reflection/refraction/absorption/etc)
        Vec3 normal;
        // Material material;
        Material *mat;
        // bool to check if the ray hit the front face or the back face of the object
        bool front_face;
        // MSAA EDGE CHECK (for antialiasing)
        bool edge_hit;

        // blank constructor
        __device__ HitRecord() : t(0), p(Vec3()), normal(Vec3()), mat(nullptr), front_face(false), edge_hit(false) {}

        // constructor with parameters
        __device__ HitRecord(float t, Vec3 p, Vec3 normal, Material *mat, bool is_front_face, bool edge_hit) : t(t), p(p), normal(normal), mat(mat), front_face(is_front_face), edge_hit(edge_hit) {}
}; 


#endif