#ifndef INTERVAL_HPP
#define INTERVAL_HPP

__device__ const double infinity = HUGE_VAL;

class interval {
  public:
    double min, max;

    __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __device__ interval(double min, double max) : min(min), max(max) {}

    __device__ double size() const {
        return max - min;
    }

    __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __device__ double clamp(double x) const {
        return (x < min) ? min : (x > max) ? max : x;
    }

    // have not been using
    // static const interval empty, universe;
};

//haven't been using
// inline const interval interval::empty    = interval(+infinity, -infinity);
// inline const interval interval::universe = interval(-infinity, +infinity);

#endif