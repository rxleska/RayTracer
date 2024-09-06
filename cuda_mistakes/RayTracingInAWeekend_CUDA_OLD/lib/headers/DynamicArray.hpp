#ifndef DYNAMIC_ARRAY_HPP
#define DYNAMIC_ARRAY_HPP

//this class is for a dynamically sized array for __device__ memory in CUDA

template <typename T>
class DynamicArray {
    private:
        T* data;
        int size;
        int capacity;

        __device__ void resize(int new_capacity);
        __device__ int get_capacity();

    public:
        // constructors
        __device__ DynamicArray();
        __device__ DynamicArray(int capacity);
        __device__ DynamicArray(T* data, int size, int capacity);
        __device__ ~DynamicArray();

        // getters
        __device__ T get(int index);
        __device__ int get_size();
        __device__ T pop_back();

        // setters
        __device__ void set(int index, T value);
        __device__ void push_back(T value);
        __device__ void clear();

        // operators
        __device__ T operator[](int index);
        __device__ DynamicArray<T>& operator=(const DynamicArray<T>& other);
};


























#endif