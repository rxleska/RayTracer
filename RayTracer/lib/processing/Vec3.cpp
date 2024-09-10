#include "headers/Vec3.hpp"
#include <cmath>

inline float Vec3::dot(const Vec3& v) const {
    return x*v.x + y*v.y + z*v.z;
}

inline Vec3 Vec3::cross(const Vec3& v) const {
    return Vec3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
}

inline float Vec3::mag() const {
    return sqrt(x*x + y*y + z*z);
}

inline float Vec3::length() const {
    return sqrt(x*x + y*y + z*z);
}

inline float Vec3::mag2() const {
    return x*x + y*y + z*z;
}

inline float Vec3::length2() const {
    return x*x + y*y + z*z;
}


// operator overloading
inline Vec3 Vec3::operator+(const Vec3& v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
}

inline Vec3 Vec3::operator-(const Vec3& v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
}

//dot product
inline Vec3 Vec3::operator*(const Vec3& v) const {
    return Vec3(x * v.x, y * v.y, z * v.z);
}

inline Vec3 Vec3::operator*(const float& v) const {
    return Vec3(x * v, y * v, z * v);
}

inline Vec3 Vec3::operator/(const float& v) const {
    return Vec3(x / v, y / v, z / v);
}

inline Vec3 Vec3::operator*(const double& v) const {
    return Vec3(x * v, y * v, z * v);
}

inline Vec3 Vec3::operator/(const double& v) const {
    return Vec3(x / v, y / v, z / v);
}

inline bool Vec3::isZero() const {
    return x == 0 && y == 0 && z == 0;
}

inline Vec3 Vec3::normalized() const {
    float m = mag();
    return Vec3(x/m, y/m, z/m);
}

inline void Vec3::normalize() {
    float m = mag();
    x /= m;
    y /= m;
    z /= m;
}