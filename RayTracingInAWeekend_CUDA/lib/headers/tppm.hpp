// tppm.hpp will convert an array of vector3s to a ppm image or an array of uint8_t to a ppm image

#ifndef TPPM_HPP
#define TPPM_HPP

#include <fstream>
#include <stdint.h>

#include "vec3.hpp"
#include "color.hpp"


void write_ppm(const char * filename, color ** image, int width, int height);

void write_ppm(const char * filename, uint8_t *** image, int width, int height);

#endif