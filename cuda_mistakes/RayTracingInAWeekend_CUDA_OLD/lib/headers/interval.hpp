#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include <limits.h>

const double infinity = HUGE_VAL;
__device__ const double d_infinity = HUGE_VAL;  // Another way to define infinity


class interval {
  public:
    double min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(double min, double max) : min(min), max(max) {}

    __host__ __device__ double size() const {
        return max - min;
    }

    __host__ __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __host__ __device__ double clamp(double x) const {
        return (x < min) ? min : (x > max) ? max : x;
    }

    // static const interval empty, universe;
};

// const interval interval::empty    = interval(+infinity, -infinity);
// const interval interval::universe = interval(-infinity, +infinity);

#endif