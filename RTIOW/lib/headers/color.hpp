// color definition of the vector class (vec3) 


#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"
#include "interval.hpp"

using color = vec3;

inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity = interval(0.0, 0.999);
    int rbyte = int(255.999 * intensity.clamp(r));
    int gbyte = int(255.999 * intensity.clamp(g));
    int bbyte = int(255.999 * intensity.clamp(b));

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}


#endif 