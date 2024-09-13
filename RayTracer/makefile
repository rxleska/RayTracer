# compiler setup 
CC = nvcc

# compiler flags:
# -arch=sm_89: specifies the architecture of the GPU 89 ADA GPUs (4070 TI)
# -arch=sm_87: specifies the architecture of the GPU 87 Ampere GPUs (3050 Laptops) (also supports sm_80 and sm_86)
# -rdc=true: enables relocatable device code (required for multiple files)
# CFLAGS = -arch=sm_89 -rdc=true
CFLAGS = -G -arch=sm_80 -rdc=true



PROCESSING = bin/Ray.o bin/Vec3.o bin/Camera.o
MATERIALS  = bin/Lambertian.o
HITTABLES  = bin/Sphere.o  bin/Scene.o

all: run 

run: bin bin/main.o $(PROCESSING) $(MATERIALS) $(HITTABLES)
	$(CC) $(CFLAGS) -o run bin/main.o $(PROCESSING) $(MATERIALS) $(HITTABLES)

bin:
	mkdir bin

bin/main.o: main.cu 
	$(CC) $(CFLAGS) -c main.cu -o bin/main.o

# Processing Objects

bin/Ray.o: lib/processing/Ray.cu lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/processing/Ray.cu -o bin/Ray.o

bin/Vec3.o: lib/processing/Vec3.cu lib/processing/headers/Vec3.hpp
	$(CC) $(CFLAGS) -c lib/processing/Vec3.cu -o bin/Vec3.o

bin/Camera.o: lib/processing/Camera.cu lib/processing/headers/Camera.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/processing/Camera.cu -o bin/Camera.o




# Material Objects

bin/Lambertian.o: lib/materials/Lambertian.cu lib/materials/headers/Lambertian.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Lambertian.cu -o bin/Lambertian.o



# Hittable Objects

bin/Sphere.o: lib/hittable/Sphere.cu lib/hittable/headers/Sphere.hpp lib/processing/headers/Ray.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Sphere.cu -o bin/Sphere.o

bin/Scene.o: lib/hittable/Scene.cu lib/hittable/headers/Scene.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Scene.cu -o bin/Scene.o



clean:
	rm -rf bin/ run

clean-win:
	del /s /q bin\ run.exe	