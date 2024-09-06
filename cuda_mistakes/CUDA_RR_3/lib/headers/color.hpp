// color definition of the vector class (vec3) 


#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"
#include "interval.hpp"

using color = vec3;

__host__ inline double linear_to_gamma(double linear_component);
__host__ void write_color(std::ostream& out, const color& pixel_color);


#endif 