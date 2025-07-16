#ifndef POLYGON_H
#define POLYGON_H

#include "../math_classes/vec3.h"
#include "../math_classes/ray.h"
#include "../math_classes/color.h"
#include "../hittable/hit_record.h"
#include "../materials/material.h"

struct polygon {
    vec3 v1; 
    vec3 v2;
    vec3 v3;
    vec3 normal; // Normal vector of the polygon
    material mat; // Material of the polygon
};


__device__ inline bool hit(const polygon self, ray& r, float t_min, float t_max, hit_record & record, ray& scattered, curandState* curandState) {
    // Calculate the plane equation of the polygon
    vec3 edge1 = self.v2 - self.v1;
    vec3 edge2 = self.v3 - self.v1;
    vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);
    if (fabs(a) < 1e-6) {
        return false; // Ray is parallel to the polygon
    }
    float f = 1.0f / a;
    vec3 s = r.origin - self.v1;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false; // Intersection is outside the polygon
    }
    vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false; // Intersection is outside the polygon
    }
    // Calculate the intersection point
    float t = f * dot(edge2, q);
    if (t < t_min || t > t_max) {
        return false; // Intersection is outside the valid range
    }
    record.is_front_face = dot(r.direction, self.normal) < 0.0f; 
    if(!record.is_front_face){
        return false;
    }
    record.t = t; // Set the hit distance
    record.intersection_point = r.origin + r.direction * t;
    record.normal = self.normal; 

    record.mat = self.mat; 
    return true; // Intersection occurred
}

__host__ void invert_polygon(polygon &self){
    self.normal = -self.normal;
}



#endif // POLYGON_H