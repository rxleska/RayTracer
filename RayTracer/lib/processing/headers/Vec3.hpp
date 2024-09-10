#ifndef VEC3_HPP
#define VEC3_HPP
// Pretty Standard 3d Vector Class (most functions are going to be inline for performance, 
// we don't really care about the size of the binary what we want is fast running times and less context switches)


// currently as things stand I will be using floats instead of doubles for the sake of performance no need to be way too accurate

class Vec3{
    public: 
        float x;
        float y;
        float z;

        Vec3(): x(0), y(0), z(0) {}
        Vec3(float x, float y, float z): x(x), y(y), z(z) {}

        // dot product
        inline float dot(const Vec3& v) const;
        // cross product
        inline Vec3 cross(const Vec3& v) const;

        // magnitude
        inline float mag() const;
        // length
        inline float length() const;
        // magnitude squared
        inline float mag2() const;
        // length squared
        inline float length2() const;

        // operator overloading

        inline Vec3 Vec3::operator+(const Vec3& v) const;
        inline Vec3 Vec3::operator-(const Vec3& v) const;
        inline Vec3 Vec3::operator*(const Vec3& v) const;
        inline Vec3 Vec3::operator*(const float& v) const;
        inline Vec3 Vec3::operator/(const float& v) const;
        inline Vec3 Vec3::operator*(const double& v) const;
        inline Vec3 Vec3::operator/(const double& v) const;

        // check if the vector is zero (close to zero)
        inline bool isZero() const;
        // return the normalized version of the vector
        inline Vec3 normalized() const;
        // normalize the vector in place
        void normalize();
};



#endif