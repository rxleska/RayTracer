// tppm.hpp will convert an array of vector3s to a ppm image or an array of uint8_t to a ppm image

#ifndef TPPM_HPP
#define TPPM_HPP

#include <fstream>

#include "vec3.hpp"
#include "color.hpp"


void write_ppm(const char * filename, color ** image, int width, int height){  // note that the image is stored in a 2D array of vec3s [height][width]
    std::ofstream file;
    file.open(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height-1; j >= 0; j--){
        for (int i = 0; i < width; i++){
            write_color(file, image[j][i]);
        }
    }
    file.close();

}

void write_ppm(const char * filename, uint8_t *** image, int width, int height){ // note that the image is stored in a 3d array of uint8_t [height][width][rgb]
    std::ofstream file;
    file.open(filename);
    file << "P6\n" << width << " " << height << "\n255\n";
    // due to usage of uint8_t, we can directly write the image to the file
    file.write((char *)image[0][0], width*height*3);
    file.close();
}

#endif