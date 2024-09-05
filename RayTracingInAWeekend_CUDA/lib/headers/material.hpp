#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "hittable.hpp"

enum MaterialType{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    LIGHT
};

class material
{
public:
    virtual ~material() = default;

    __host__ __device__ virtual int scatter( // 0:no scatter, 1:scatter, 2:scatter and absorb
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const;
    __host__ __device__ bool getIsLightSrc() const;
    __host__ __device__ void setIsLightSrc(bool isLightSrc);
    __host__ __device__ MaterialType getType() const;
    __host__ __device__ void setType(MaterialType type);

private:
    bool isLightSrc = false;
    MaterialType type;

};

// defining some materials

// Lambertian material - scatters light in all directions
class lambertian : public material
{
public:
    __host__ __device__ lambertian(const color &albedo) : albedo(albedo) {
        setType(LAMBERTIAN);
    }

    __host__ __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    color albedo;
};

// Metal material - scatters light in a single direction with reflection
class metal : public material
{
public:
    __host__ __device__ metal(const color &albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {
        setType(METAL);
    }

    __host__ __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    color albedo;
    double fuzz;
};

// Dielectric material - scatters light in a single direction with refraction
class dielectric : public material
{
public:
    __host__ __device__ dielectric(double refraction_index) : refraction_index(refraction_index) {
        setType(DIELECTRIC);
    }

    __host__ __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    __host__ __device__ static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        double r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};


class light : public material
{   
    public:
        __host__ __device__ light(const color &emission) : emission(emission) {setIsLightSrc(true); setType(LIGHT);}

        __host__ __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
            const override;

        __host__ __device__ color emitted() const;
    private:
        color emission;

};

#endif