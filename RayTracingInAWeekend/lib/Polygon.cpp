
#include "headers/Polygon.hpp"


bool polygon::hit(const ray& r, interval ray_t, hit_record& rec) const {
    //

    vec3 normal = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
    

    //check that the ray is not parallel to the plane -might be optimal to just remove this check
    if (dot(normal, r.direction()) == 0) { //perpendicular to normal
        return false; 
    }


    // float d = -dot(normal, r.origin());
    // float t = -(dot(normal, r.origin() / dot(normal, r.direction())));
    float t = dot(normal, vertices[0] - r.origin()) / dot(normal, r.direction());
    point3 p = r.at(t); 

    vec3 na = cross((vertices[2] - vertices[1]), (p - vertices[1])); 
    vec3 nb = cross((vertices[0] - vertices[2]), (p - vertices[2]));
    vec3 nc = cross((vertices[1] - vertices[0]), (p - vertices[0]));
    float aa = 0.5 * na.length();
    float ab = 0.5 * nb.length();
    float ac = 0.5 * nc.length();
    float a = aa + ab + ac;

    float triangleArea = normal.length() / 2;
    if (a > triangleArea +  (triangleArea * 0.0001)) {
        // std::cout << p.x() << " " << p.y() << " " << p.z() << std::endl;
        // if(p.x() < 5 && p.x() > -5 && p.y() < 5 && p.y() > -5 && p.z() < -2 && p.z() > -4){
        //     std::cout << a << ":" << triangleArea << std::endl;
        // }

        return false;
    }
    if(!ray_t.surrounds(t)) {
        return false;
    }

    // if angle between ray and normal is greater than 90 degrees log error
    if (dot(normal, r.direction()) > 0) {
        return false;
    }

    rec.t = t;
    rec.p = p;
    rec.set_face_normal(r, normal);
    rec.mat = mat;
    return true;    
}