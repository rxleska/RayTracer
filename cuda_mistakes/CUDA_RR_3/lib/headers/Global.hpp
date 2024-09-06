#ifndef GLOBAL_HPP
#define GLOBAL_HPP
#include <curand_kernel.h>
#include <iostream>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

#endif