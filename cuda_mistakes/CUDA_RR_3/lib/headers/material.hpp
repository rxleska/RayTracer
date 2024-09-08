#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "hittable.hpp"

class material
{
public:
    __device__ virtual ~material() = default;

    __device__ virtual int scatter( // 0:no scatter, 1:scatter, 2:scatter and absorb
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const;
    __device__ bool getIsLightSrc() const;
    __device__ void setIsLightSrc(bool isLightSrc);

private:
    bool isLightSrc = false;
};

// defining some materials

// Lambertian material - scatters light in all directions
class lambertian : public material
{
public:
    __device__ lambertian(const color &albedo) : albedo(albedo) {}

    __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    color albedo;
};

// Metal material - scatters light in a single direction with reflection
class metal : public material
{
public:
    __device__ metal(const color &albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    color albedo;
    double fuzz;
};

// Dielectric material - scatters light in a single direction with refraction
class dielectric : public material
{
public:
    __device__ dielectric(double refraction_index) : refraction_index(refraction_index) {}

    __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
        const override;

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;

    __device__ static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};


class light : public material
{   
    public:
        __device__ light(const color &emission) : emission(emission) {setIsLightSrc(true);}

        __device__ int scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
            const override;

        __device__ color emitted() const;
    private:
        color emission;

};

#endif