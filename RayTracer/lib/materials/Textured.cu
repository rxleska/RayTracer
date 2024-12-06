#include "headers/Textured.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


// Treated as a Lambertian material, but with a texture, albedo is the texture
__device__ int Textured::scatter(const Ray &ray_in, HitRecord &rec, Vec3 &attenuation, Ray &scattered_out, curandState * rand_state) const {
    Vec3 normal = rec.normal; // get the normal of the hit point

    // get a random unit vector
    Vec3 bounceMod = Vec3::random(-10,10,rand_state); // 10 is arbitrary, since we are normalizing it later
    bounceMod.make_unit();

    // get the new direction
    Vec3 target = normal + bounceMod;
    
    // degenerate case where the new direction is close to zero
    if (target.isZero()) {
        target = normal;
    }

    //create new ray
    scattered_out = Ray(rec.p, target);

    //set the attenuation (color modification)

    // get the texture color
    int u = int(width * rec.u);
    int v = int(height * (1 - rec.v));
    
    attenuation = texture[u + width * v];
    return 1;
}

// #include <fstream>
// #include <sstream>
// #include <iostream>
// #include <string>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

__host__ float* load_texture(const char* filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary); // Open in binary mode for P6
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    std::getline(file, line);
    if (line != "P3" && line != "P6") {
        std::cerr << "Error: File is not in P3 or P6 format" << std::endl;
        return nullptr;
    }
    bool isBinary = (line == "P6");

    // Skip comments and read width and height
    do {
        std::getline(file, line);
    } while (!line.empty() && line[0] == '#');

    std::stringstream ss(line);
    ss >> width >> height;
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid width or height" << std::endl;
        return nullptr;
    }

    // Skip comments and read max color value
    int max_color_value;
    do {
        std::getline(file, line);
    } while (!line.empty() && line[0] == '#');

    ss.clear();
    ss.str(line);
    ss >> max_color_value;
    if (max_color_value <= 0 || max_color_value > 65535) {
        std::cerr << "Error: Invalid max color value" << std::endl;
        return nullptr;
    }

    // Allocate memory for texture
    float* texture = new float[width * height * 3];

    if (isBinary) {
        // P6: Binary format
        unsigned char r, g, b;
        for (int i = 0; i < width * height; ++i) {
            file.read(reinterpret_cast<char*>(&r), 1);
            file.read(reinterpret_cast<char*>(&g), 1);
            file.read(reinterpret_cast<char*>(&b), 1);

            texture[i * 3 + 0] = r / static_cast<float>(max_color_value);
            texture[i * 3 + 1] = g / static_cast<float>(max_color_value);
            texture[i * 3 + 2] = b / static_cast<float>(max_color_value);
        }
    } else {
        // P3: ASCII format
        for (int i = 0; i < width * height; ++i) {
            int r, g, b;
            file >> r >> g >> b;

            texture[i * 3 + 0] = r / static_cast<float>(max_color_value);
            texture[i * 3 + 1] = g / static_cast<float>(max_color_value);
            texture[i * 3 + 2] = b / static_cast<float>(max_color_value);
        }
    }

    if (!file) {
        std::cerr << "Error: File read error" << std::endl;
        delete[] texture;
        return nullptr;
    }

    return texture;
}


// //loads only ppms in the format P3
// __host__ float *load_texture(const char *filename, int &width, int &height) {
//     std::ifstream file(filename);
//     std::string line;
//     std::getline(file, line);
//     if (line != "P3" && line != "P6") {
//         std::cerr << "Error: File is not in P3 or P6 format" << std::endl;
//         return nullptr;
//     }

//     // Skip comments and read width and height
//     std::getline(file, line);
//     while (line[0] == '#') {
//         std::getline(file, line);
//     }
//     std::stringstream ss(line);
//     ss >> width >> height;

//     // Skip the max color value
//     std::getline(file, line);

//     // Allocate memory for texture
//     float *texture = new float[width * height * 3];

//     for (int i = 0; i < width * height; i++) {
//         unsigned char r, g, b;
//         if (line == "P3") {
//             file >> r >> g >> b;
//         } else { // P6
//             file.read(reinterpret_cast<char*>(&r), 1);
//             file.read(reinterpret_cast<char*>(&g), 1);
//             file.read(reinterpret_cast<char*>(&b), 1);
//         }
//         texture[i * 3 + 0] = r / 255.0f;
//         texture[i * 3 + 1] = g / 255.0f;
//         texture[i * 3 + 2] = b / 255.0f;
//     }

//     return texture;
// }
