#include "headers/material.hpp"

// Material class


// Base Scatter function for materials
// 0:no scatter, 1:scatter, 2:scatter and absorb
int material::scatter( 
    const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const
{
    return false;
}

// Get if the material is a light source (therefore absorbs inverse rays)
bool material::getIsLightSrc() const { return isLightSrc; }

// Set if the material is a light source (therefore absorbs inverse rays)
void material::setIsLightSrc(bool isLightSrc) { this->isLightSrc = isLightSrc; }

// defining some materials

// Lambertian material - scatters light in all directions

// Lambertian material Scatter , scatters light in all directions (randomly) creating a Matte finish
int lambertian::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    //ensure rec.normal is a unit vector
    vec3 norm = unit_vector(rec.normal);

    auto scatter_direction = norm + random_unit_vector();

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = norm;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return 1;
}

// Metal material - scatters light in a single direction with reflection

// Metal material Scatter, scatters light in a single direction with reflection creating a shiny finish with fuzziness
int metal::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0 ? 1 : 0);
}

// Dielectric material - scatters light in a single direction with refraction


// Dielectric material Scatter, scatters light in a single direction with refraction creating a glassy finish
int dielectric::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    attenuation = color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > random_double())
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, direction);
    return 1;
}


// Light material - scatters light in all directions (inverse: absorbs inverse rays)

// Light material Scatter, scatters light in all directions (inverse: absorbs inverse rays)
int light::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered)
    const 
{
    return 2;
}

// Light material emitted color
color light::emitted() const { return emission; }