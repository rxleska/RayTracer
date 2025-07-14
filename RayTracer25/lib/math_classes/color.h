#ifndef COLOR_H
#define COLOR_H

#include "vec3.h" // Include the vec3 header for color representation

using color = vec3; // Define color as a vec3 for RGB colors

__host__ __device__ inline color new_color(float r, float g, float b) {
    return {r, g, b};
}

struct color_uint8 {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    __device__ inline color_uint8(uint8_t red, uint8_t green, uint8_t blue) : r(red), g(green), b(blue) {}
};


__device__ inline color_uint8 color_to_uint8(const color& c) {
    // Convert color to uint8_t array (RGB)
    return color_uint8(
        static_cast<uint8_t>(fminf(fmaxf(c.x * 255.0f, 0.0f), 255.0f)),
        static_cast<uint8_t>(fminf(fmaxf(c.y * 255.0f, 0.0f), 255.0f)),
        static_cast<uint8_t>(fminf(fmaxf(c.z * 255.0f, 0.0f), 255.0f))
    );
}

#endif // COLOR_H