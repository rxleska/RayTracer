#include "headers/DynamicArray.hpp"


// constructors
template <typename T>
__device__ DynamicArray<T>::DynamicArray() {
    this->size = 0;
    this->capacity = 1;
    this->data = cudaMalloc(sizeof(T) * this->capacity);
}

template <typename T>
__device__ DynamicArray<T>::DynamicArray(int capacity) {
    this->size = 0;
    this->capacity = capacity;
    this->data = cudaMalloc(sizeof(T) * this->capacity);
}

template <typename T>
__device__ DynamicArray<T>::DynamicArray(T* data, int size, int capacity){
    this->size = size;
    this->capacity = capacity;
    this->data = cudaMalloc(sizeof(T) * this->capacity);
    cudaMemcpy(this->data, data, sizeof(T) * this->size, cudaMemcpyHostToDevice);
}


template <typename T>
__device__ DynamicArray<T>::~DynamicArray() {
    cudaFree(this->data);
}



// getters
template <typename T>
__device__ T DynamicArray<T>::get(int index) {
    return this->data[index];
}

template <typename T>
__device__ int DynamicArray<T>::get_size() {
    return this->size;
}

// setters
template <typename T>
__device__ void DynamicArray<T>::set(int index, T value) {
    // will not allow to set value if index is out of bounds (design choice)
    if(index >= this->size) {
        return;
    }
    this->data[index] = value;
}

template <typename T>
__device__ void DynamicArray<T>::push_back(T value) {
    if(this->size == this->capacity) {
        this->capacity *= 2;
        T* new_data = cudaMalloc(sizeof(T) * this->capacity);
        cudaMemcpy(new_data, this->data, sizeof(T) * this->size, cudaMemcpyDeviceToDevice);
        cudaFree(this->data);
        this->data = new_data;
    }
    this->data[this->size] = value;
    this->size++;
}

template <typename T>
__device__ T DynamicArray<T>::pop_back() {
    if(this->size == 0) {
        return;
    }
    this->size--;
    return this->data[this->size];
    //no reason to delete the data due to previous design choice
}


template <typename T>
__device__ void DynamicArray<T>::clear(){
    this->size = 0;
}


//operators 
template <typename T>
__device__ T DynamicArray<T>::operator[](int index) {
    return this->data[index];
}

template <typename T>
__device__ DynamicArray<T>& DynamicArray<T>::operator=(const DynamicArray<T>& other) {
    this->size = other.size;
    this->capacity = other.capacity;
    cudaFree(this->data);
    this->data = cudaMalloc(sizeof(T) * this->capacity);
    cudaMemcpy(this->data, other.data, sizeof(T) * this->size, cudaMemcpyDeviceToDevice);
    return *this;
}