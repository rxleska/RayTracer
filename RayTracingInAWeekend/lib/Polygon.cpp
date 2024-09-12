
#include "headers/Polygon.hpp"

// hit function for polygon class
bool polygon::hit(const ray& r, interval ray_t, hit_record& rec) const {
    //create normal vector for the plane containing the polygon 
    vec3 normal = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
    
    // if angle between ray and normal is greater than 90 degrees return miss (backface culling (culling is the process of discarding objects that are not visible to the camera, we don't want to render the back of the polygons))
    if (dot(normal, r.direction()) >= -0.00001) {
        return false;
    }

    //calculate the point of intersection
    float t = dot(normal, vertices[0] - r.origin()) / dot(normal, r.direction());
    point3 p = r.at(t); 

    
    // if hit position is not in the current ray interval return false
    if(!ray_t.surrounds(t)) {
        return false;
    }

    // Calculate the area of the 3 triangles formed by the intersection point and the vertices of the polygon
    vec3 na = cross((vertices[2] - vertices[1]), (p - vertices[1])); 
    vec3 nb = cross((vertices[0] - vertices[2]), (p - vertices[2]));
    vec3 nc = cross((vertices[1] - vertices[0]), (p - vertices[0]));
    float aa = 0.5 * na.length();
    float ab = 0.5 * nb.length();
    float ac = 0.5 * nc.length();
    float a = aa + ab + ac;

    // if the area of the 3 triangles is greater than the area of the polygon hit is a miss
    float triangleArea = normal.length() / 2;
    if (a > triangleArea +  (triangleArea * 0.0001)) {
        return false;
    }

    // if the hit is valid set the hit record and return true
    rec.t = t; //time of intersection
    rec.p = p; //point of intersection
    rec.set_face_normal(r, normal); //normal vector
    rec.mat = mat; //material
    return true;    
}