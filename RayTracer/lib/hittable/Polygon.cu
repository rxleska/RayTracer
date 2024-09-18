
#include "headers/Polygon.hpp"
#include "assert.h"

#ifndef F_EPSILON
#define F_EPSILON 0.0001f
#endif


__device__ Polygon::Polygon(Vec3 * vertices, int num_vertices, Material * mat): vertices(vertices), num_vertices(num_vertices), mat(mat) {
    if(num_vertices < 3){
        num_vertices = 0;
        assert(false);
    }

    if(!is_coplanar()){
        num_vertices = 0;
        assert(false);
    }

    calculate_normal_and_area();
}


// function to check if a ray hits the sphere
__device__ bool Polygon::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    //check if the ray is parallel to the plane
    float denom = normal.dot(r.direction);
    if(denom > -F_EPSILON){
        return false;
    }

    //calculate the intersection point
    float t = (vertices[0] - r.origin).dot(normal) / denom;
    if(t < t_min || t > t_max){
        return false;
    }

    Vec3 p = r.pointAt(t);

    //check if the point is inside the polygon by calculating the area of the polygon formed by the point and the vertices
    bool edge_hit = false;
    float totalArea = 0.0f;
    for(int i = 0; i < num_vertices; i++){
        // Vec3 v1 = vertices[i] - p;
        // Vec3 v2 = vertices[(i+1)%num_vertices] - p;
        Vec3 v1 = vertices[(i+1)%num_vertices] - vertices[i];
        Vec3 v2 = p - vertices[i];
        float a = 0.5f * v1.cross(v2).length();
        if(a < area*0.01f){
            edge_hit = true;
        }
        totalArea += a;
    }
    if(totalArea > area + area*F_EPSILON){
        return false;
    }
    
    rec.t = t;
    rec.p = p;
    rec.normal = normal;
    rec.front_face = true; //wouldn't have hit the polygon if it was not facing the front
    rec.mat = mat;
    rec.edge_hit = edge_hit;
    return true;
}

#ifndef F_MAX
#define F_MAX 3.402823466e+38F
#endif

__device__ void Polygon::getBounds(float& x_min, float& x_max, float& y_min, float& y_max, float& z_min, float& z_max) const{
    // get float max and min
    x_min = y_min = z_min = F_MAX;
    x_max = y_max = z_max = -F_MAX;
    for(int i = 0; i < num_vertices; i++){
        x_min = fminf(x_min, vertices[i].x);
        x_max = fmaxf(x_max, vertices[i].x);
        y_min = fminf(y_min, vertices[i].y);
        y_max = fmaxf(y_max, vertices[i].y);
        z_min = fminf(z_min, vertices[i].z);
        z_max = fmaxf(z_max, vertices[i].z);
    }
}

__device__ bool Polygon::is_coplanar() const{
    Vec3 currentCross;
    for (int i = 0; i < num_vertices - 3; i++) {
        Vec3 v1 = vertices[i + 1] - vertices[i];
        Vec3 v2 = vertices[i + 2] - vertices[i];
        Vec3 normal = v1.cross(v2).normalized();
        if (i == 0) {
            currentCross = normal;  // store the first cross product
        } else {
            if (fabs(currentCross.dot(normal)) > F_EPSILON) { // or use fabs(currentCross.dot(normal)) > F_EPSILON
                return false;
            }
        }
    }
    return true;
}

__device__ void Polygon::calculate_normal_and_area(){
    Vec3 v1 = vertices[1] - vertices[0];
    Vec3 v2 = vertices[num_vertices-1] - vertices[0];
    normal = v1.cross(v2);

    // calculate the area of the polygon
    //choose center of polygon as reference point
    Vec3 center = Vec3(0,0,0);
    for(int i = 0; i < num_vertices; i++){
        center = center + vertices[i];
    }
    center = center / float(num_vertices);

    //for each triangle in the polygon calculate 
    area = 0.0f;
    for(int i = 0; i < num_vertices; i++){
        Vec3 v1 = vertices[i] - center;
        Vec3 v2 = vertices[(i+1)%num_vertices] - center;
        area += 0.5f * v1.cross(v2).length();
    }

    //normalize the normal
    normal = normal.normalized();
}


__device__ Polygon * Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Material * mat){
    Vec3 * vertices = new Vec3[3];
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    return new Polygon(vertices, 3, mat);
}

__device__ Polygon * Quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Material * mat){
    Vec3 * vertices = new Vec3[4];
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    vertices[3] = v4;
    return new Polygon(vertices, 4, mat);
}