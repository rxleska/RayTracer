#ifndef BOX_HPP
#define BOX_HPP

#include "Hitable.hpp"

//class to represent a box in the scene

class Box: public Hitable{
    public:
        Vec3 min, max;
        Material * mat;

        __device__ Box(Vec3 min, Vec3 max, Material *mat) : min(min), max(max), mat(mat) {
            if(min.x > max.x){
                float temp = min.x;
                min.x = max.x;
                max.x = temp;
            }
            if(min.y > max.y){
                float temp = min.y;
                min.y = max.y;
                max.y = temp;
            }
            if(min.z > max.z){
                float temp = min.z;
                min.z = max.z;
                max.z = temp;
            }
        }

        __device__ float bound(int axis, int side) const;

        // function to check if a ray hits the sphere
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const override;

        // function to get the bounding box of the sphere
        __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const override;
        
        __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const override;

        __device__ virtual  void debug_print() const override;

        __device__ virtual Vec3 getRandomPointInHitable(curandState *state) const override;

        __device__ virtual float get2dArea() const override;

};

#endif