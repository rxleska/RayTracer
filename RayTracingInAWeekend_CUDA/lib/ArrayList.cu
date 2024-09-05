#include "headers/ArrayList.hpp"

// Constructor
template <typename T>
__host__ __device__ ArrayList<T>::ArrayList(size_t initial_capacity)
    : size_(0), capacity_(initial_capacity) {
    if (capacity_ > 0) {
        checkCudaErrors(cudaMalloc(&data_, capacity_ * sizeof(T)));
    } else {
        data_ = nullptr;
    }
}

// Destructor
template <typename T>
__host__ __device__ ArrayList<T>::~ArrayList() {
    if (data_ != nullptr) {
        checkCudaErrors(cudaFree(data_));
    }
}

// Copy Constructor
template <typename T>
__host__ __device__ ArrayList<T>::ArrayList(const ArrayList& other)
    : size_(other.size_), capacity_(other.capacity_) {
    if (capacity_ > 0) {
        checkCudaErrors(cudaMalloc(&data_, capacity_ * sizeof(T)));
        arraylist_cudaMemcpyDeviceToDevice(data_, other.data_, size_);
    } else {
        data_ = nullptr;
    }
}

// Move Constructor
template <typename T>
__host__ __device__ ArrayList<T>::ArrayList(ArrayList&& other) noexcept
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

// Copy Assignment Operator
template <typename T>
__host__ __device__ ArrayList<T>& ArrayList<T>::operator=(const ArrayList& other) {
    if (this != &other) {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        size_ = other.size_;
        capacity_ = other.capacity_;
        if (capacity_ > 0) {
            checkCudaErrors(cudaMalloc(&data_, capacity_ * sizeof(T)));
            arraylist_cudaMemcpyDeviceToDevice(data_, other.data_, size_);
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

// Move Assignment Operator
template <typename T>
__host__ __device__ ArrayList<T>& ArrayList<T>::operator=(ArrayList&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            checkCudaErrors(cudaFree(data_));
        }
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

// Add Element
template <typename T>
__host__ __device__ void ArrayList<T>::add(const T& element) {
    if (size_ >= capacity_) {
        resize(2 * capacity_);
    }
    // Since we are adding elements, use cudaMemcpy for the host to device transfer
    arraylist_cudaMemcpyHostToDevice(data_ + size_, &element, 1);
    ++size_;
}

// Get Element
template <typename T>
__host__ __device__ T ArrayList<T>::get(size_t index) const {
    T value;
    arraylist_cudaMemcpyDeviceToHost(&value, data_ + index, 1);
    return value;
}

// Size
template <typename T>
__host__ __device__ size_t ArrayList<T>::size() const {
    return size_;
}

// Capacity
template <typename T>
__host__ __device__ size_t ArrayList<T>::capacity() const {
    return capacity_;
}

// Clear
template <typename T>
__host__ __device__ void ArrayList<T>::clear() {
    size_ = 0;
}

// Resize the underlying storage
template <typename T>
__host__ __device__ void ArrayList<T>::resize(size_t new_capacity) {
    T* new_data;
    checkCudaErrors(cudaMalloc(&new_data, new_capacity * sizeof(T)));
    arraylist_cudaMemcpyDeviceToDevice(new_data, data_, size_);
    cudaFree(data_);
    data_ = new_data;
    capacity_ = new_capacity;
}


#include "headers/hittable.hpp"

template class ArrayList<hittable *>;

#include <thread>

template class ArrayList<std::thread>;