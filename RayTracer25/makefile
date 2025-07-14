CC = nvcc

# CFLAGS = -arch=sm_89 -rdc=true -use_fast_math -O3 -ftz=true -prec-div=false -prec-sqrt=false -I/usr/include/SDL2 -D_REENTRANT -lSDL2
# arch = sm_89 : architecture for my 4070
# rdc=true : enables device code linking (required for mulitple CUDA files)
# use_fast_math : enables fast math operations (less precision)
# 03 : optimization level (max optimization)
# ftz=true : flush denormalized numbers to zero
# prec-div=false : disables precise division
# prec-sqrt=false : disables precise square root
# I/usr/include/SDL2 : include path for SDL2 headers
# D_REENTRANT : defines the macro for multithreading support
# -lSDL2 : links against the SDL2 library


# simplified version for new development (may add more flags later)
CFLAGS = -arch=sm_89 #-rdc=true  

hittable_headers = lib/hittable/sphere.h lib/hittable/hittable.h lib/hittable/hit_record.h
math_headers = lib/math_classes/vec3.h lib/math_classes/color.h lib/math_classes/ray.h
base_headers = lib/camera.h lib/kernel_code.h lib/cuda_check_funcs.h lib/random_cuda_funcs.h
headers = $(math_headers) $(hittable_headers) $(base_headers)


# makefile default target
all: run

# compile the run executable
run: bin bin/main.obj 
	$(CC) $(CFLAGS) -o run bin/main.obj 

bin:
	mkdir bin 

bin/main.obj: main.cu $(headers)
	$(CC) $(CFLAGS) -c main.cu -o bin/main.obj


# clean up generated files
clean:
	rm -rf bin/ run

rebuild: clean all

san: run
	compute-sanitizer --force-blocking-launches --generate-coredump ./run

initcheck: run 
	compute-sanitizer --tool initcheck ./run

memcheck: run
	compute-sanitizer --tool memcheck ./run

racecheck: run
	compute-sanitizer --tool racecheck ./run

synccheck: run
	compute-sanitizer --tool synccheck ./run