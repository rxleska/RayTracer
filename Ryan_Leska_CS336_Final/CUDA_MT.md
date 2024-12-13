This document outlines my understanding of my use of CUDA for rendering

### calling global

When a global function is called it is passed 2-4 parameters 

```cpp
EXAMPLE_GLOBAL<<<BLOCKS, THREADS>>>
```

Where BLOCKS is the number of blocks (1-3 dimensionally defined) and THREADS is the number of threads per block (1-3 dimensionally defined). Every Thread is ran on each block. In the global code the current block and thread index can be found using the following code:

```cpp
int threadX = threadIdx.x;
int threadY = threadIdx.y;
int threadZ = threadIdx.z;

int blockX = blockIdx.x;
int blockY = blockIdx.y;
int blockZ = blockIdx.z;

int total x = threadX + blockX * blockDim.x;
int total y = threadY + blockY * blockDim.y;
int total z = threadZ + blockZ * blockDim.z;
```

Since I am working with 2D picture as my output I am using a 2d block and thread setup.

### thread and block sizing

I still don't completely understand the best way to define blocks and threads, but this is what I have read.

Threads on a gpu are put together into blocks called CUDA streaming multiprocessors (SMs). 
Blocks are allocated to SMs where they are ran on the SMs threads.

From testing I have been unable to use more than 512 cores on either of my gpus (3070 mobile) or (4070 ti desktop)

Also I believe that the number of threads per SM is the amount of CUDA Cores / the number of SMs. Also I believe that cuda cores are the same as shading units on a gpu. This idea is not supported by the math in the table in the GPU TABLE section. The table would imply that given I was able to assign 512 threads many things could be happening.

1. Multiple SMs can be ran on a single block
2. The number of threads per SM is not the number of CUDA cores
3. SMs will uses its 128 threads to run the 512 threads assigned to it by each thread picking up the work of 4 threads (in order)
4. Blocks are not ran on SMs but jobs from blocks are sent to SMs to be ran?
5. Something else ...

### GPU TABLES

| GPU             | SMs | CUDA Cores | Threads per SM | FP32 TFLOPS | TENSOR CORES |
|-----------------|-----|------------|----------------|-------------|--------------|
| 3070 Mobile     | 40  | 5120       | 128            | 15.97       | 160          |
| 4070 Ti Desktop | 60  | 7680       | 128            | 40.9        | 240          |

| GPU             | ARCH         | nvcc flag                           |
|-----------------|--------------|-------------------------------------|
| 3070 Mobile     | Ampere       | -arch=sm_80 -arch=sm_86 -arch=sm_87 |
| 4070 Ti Desktop | Ada Lovelace | -arch=sm_89                         |