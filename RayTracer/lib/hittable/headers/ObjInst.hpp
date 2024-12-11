#ifndef OBJ_INST_HPP
#define OBJ_INST_HPP

#include "Hitable.hpp"

// This is a subclass of Hitable that abstracts the rotation and translation of an object

// abstraction that only allows rotation
class ObjInstRot : public Hitable {
    public:
    Hitable *obj;
    Vec3 rot;

    __device__ ObjInstRot(Hitable *obj, Vec3 rotation) : obj(obj), rot(rotation) {}
    __device__ ObjInstRot(Hitable *obj, float z_axis_rot) : obj(obj), rot(Vec3(0,0,z_axis_rot)) {}

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const override;

    __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const override;
        
    __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const override;

    __device__ virtual  void debug_print() const override;

    __device__ virtual Vec3 getRandomPointInHitable(curandState *state) const override;

    __device__ virtual float get2dArea() const override;

};

// abstraction that only allows translation
class ObjInstTrans : public Hitable {
    public: 
        Hitable *obj;
        Vec3 translation;

    __device__ ObjInstTrans(Hitable *obj, Vec3 translation) : obj(obj), translation(translation) {}

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const override;

    __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const override;
        
    __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const override;

    __device__ virtual  void debug_print() const override;

    __device__ virtual Vec3 getRandomPointInHitable(curandState *state) const override;

    __device__ virtual float get2dArea() const override;

};

// abstraction that only allows scaling
// MAY BE ADDED LATER
// class ObjInstScale : public ObjInst {
//     __device__ ObjInstScale(Hitable *obj, Vec3 scale) : ObjInst(obj, Vec3(), Vec3(), scale) {}
//     __device__ ObjInstScale(Hitable *obj, float scale) : ObjInst(obj, Vec3(), Vec3(), Vec3(scale,scale,scale)) {}

//     __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, curandState *state) const override;

//     __device__ virtual void getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const override;
        
//     __device__ virtual bool insideBox(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) const override;

//     __device__ virtual  void debug_print() const override;

//     __device__ virtual Vec3 getRandomPointInHitable(curandState *state) const override;

//     __device__ virtual float get2dArea() const override;
// };



#endif