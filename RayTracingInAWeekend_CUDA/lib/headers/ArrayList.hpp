#ifndef DEVICE_ARRAYLIST_HPP
#define DEVICE_ARRAYLIST_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "Global.hpp"

template <typename T>
class ArrayList {
public:
    // Constructors and Destructor
    __host__ __device__ ArrayList(size_t initial_capacity = 10);
    __host__ __device__ ~ArrayList();

    // Copy and Move Constructors and Assignment Operators
    __host__ __device__ ArrayList(const ArrayList& other);
    __host__ __device__ ArrayList(ArrayList&& other) noexcept;
    __host__ __device__ ArrayList& operator=(const ArrayList& other);
    __host__ __device__ ArrayList& operator=(ArrayList&& other) noexcept;

    // Methods
    __host__ __device__ void add(const T& element);
    __host__ __device__ T get(size_t index) const;
    __host__ __device__ size_t size() const;
    __host__ __device__ size_t capacity() const;
    __host__ __device__ void clear();

private:
    T* data_;
    size_t size_;
    size_t capacity_;

    __host__ __device__ void resize(size_t new_capacity);
};

// Helper functions for device code
template <typename T>
__host__ __device__ void arraylist_cudaMemcpyHostToDevice(T* dst, const T* src, size_t count) {
    checkCudaErrors(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
__host__ __device__ void arraylist_cudaMemcpyDeviceToHost(T* dst, const T* src, size_t count) {
    checkCudaErrors(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
__host__ __device__ void arraylist_cudaMemcpyDeviceToDevice(T* dst, const T* src, size_t count) {
    checkCudaErrors(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}


#endif // DEVICE_ARRAYLIST_HPP
