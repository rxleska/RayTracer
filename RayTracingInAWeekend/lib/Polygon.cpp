
#include "headers/Polygon.hpp"


bool polygon::hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 normal = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
    // float d = -dot(normal, r.origin());
    float t = -(dot(normal, r.origin() / dot(normal, r.direction())));
    point3 p = r.at(t); 

    vec3 na = cross((vertices[2] - vertices[1]), (p - vertices[1])); 
    vec3 nb = cross((vertices[0] - vertices[2]), (p - vertices[2]));
    vec3 nc = cross((vertices[1] - vertices[0]), (p - vertices[0]));
    float aa = 0.5 * na.length();
    float ab = 0.5 * nb.length();
    float ac = 0.5 * nc.length();
    float a = aa + ab + ac;

    float triangleArea = normal.length() / 2;
    if (a > triangleArea) {
        return false;
    }
    if(!ray_t.surrounds(t)) {
        return false;
    }


    rec.t = t;
    rec.p = p;
    rec.set_face_normal(r, normal);
    rec.mat = mat;
    return true;    
}