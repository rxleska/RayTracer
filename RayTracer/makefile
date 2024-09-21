# compiler setup 
CC = nvcc

# compiler flags:
# -arch=sm_89: specifies the architecture of the GPU 89 ADA GPUs (4070 TI)
# -arch=sm_87: specifies the architecture of the GPU 87 Ampere GPUs (3050 Laptops) (also supports sm_80 and sm_86)
# -rdc=true: enables relocatable device code (required for multiple files)
# CFLAGS = -arch=sm_89 -rdc=true -G
# CFLAGS = -G -arch=sm_80 -rdc=true
# CFLAGS = -arch=sm_89 -rdc=true -Xptxas -O2 -Xcompiler -O2
# CFLAGS = -arch=sm_89 -rdc=true -G
# CFLAGS = -arch=sm_80 -rdc=true
CFLAGS = -arch=sm_89 -rdc=true -G -g

# testing -G flag removes all optimizations (really slow)
# no flags defaults to all optimizations




PROCESSING = bin/Ray.obj bin/Vec3.obj bin/Camera.obj
MATERIALS  = bin/Lambertian.obj bin/Metal.obj bin/Dielectric.obj bin/Light.obj bin/LambertianBordered.obj bin/Textured.obj
HITTABLES  = bin/Sphere.obj  bin/Scene.obj bin/Polygon.obj bin/Octree.obj

all: run 

run: bin bin/main.obj $(PROCESSING) $(MATERIALS) $(HITTABLES)
	$(CC) $(CFLAGS) -o run bin/main.obj --link $(PROCESSING) $(MATERIALS) $(HITTABLES) 

bin:
	mkdir bin

bin/main.obj: main.cu lib/processing/headers/Ray.hpp lib/processing/headers/Vec3.hpp lib/processing/headers/Camera.hpp lib/materials/headers/Lambertian.hpp lib/materials/headers/Metal.hpp lib/materials/headers/Dielectric.hpp lib/materials/headers/Light.hpp lib/materials/headers/LambertianBordered.hpp lib/hittable/headers/Sphere.hpp lib/hittable/headers/Scene.hpp lib/hittable/headers/Polygon.hpp lib/hittable/headers/Octree.hpp
	$(CC) $(CFLAGS) -c main.cu -o bin/main.obj

# Processing Objects

bin/Ray.obj: lib/processing/Ray.cu lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/processing/Ray.cu -o bin/Ray.obj

bin/Vec3.obj: lib/processing/Vec3.cu lib/processing/headers/Vec3.hpp
	$(CC) $(CFLAGS) -c lib/processing/Vec3.cu -o bin/Vec3.obj

bin/Camera.obj: lib/processing/Camera.cu lib/processing/headers/Camera.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/processing/Camera.cu -o bin/Camera.obj




# Material Objects

bin/Lambertian.obj: lib/materials/Lambertian.cu lib/materials/headers/Lambertian.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Lambertian.cu -o bin/Lambertian.obj

bin/Metal.obj: lib/materials/Metal.cu lib/materials/headers/Metal.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Metal.cu -o bin/Metal.obj

bin/Dielectric.obj: lib/materials/Dielectric.cu lib/materials/headers/Dielectric.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Dielectric.cu -o bin/Dielectric.obj

bin/Light.obj: lib/materials/Light.cu lib/materials/headers/Light.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Light.cu -o bin/Light.obj

bin/LambertianBordered.obj: lib/materials/LambertianBordered.cu lib/materials/headers/LambertianBordered.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/LambertianBordered.cu -o bin/LambertianBordered.obj

bin/Textured.obj: lib/materials/Textured.cu lib/materials/headers/Textured.hpp lib/materials/headers/Material.hpp lib/processing/headers/Vec3.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/materials/Textured.cu -o bin/Textured.obj


# Hittable Objects

bin/Sphere.obj: lib/hittable/Sphere.cu lib/hittable/headers/Sphere.hpp lib/processing/headers/Ray.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Sphere.cu -o bin/Sphere.obj

bin/Scene.obj: lib/hittable/Scene.cu lib/hittable/headers/Scene.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Scene.cu -o bin/Scene.obj

bin/Polygon.obj: lib/hittable/Polygon.cu lib/hittable/headers/Polygon.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Polygon.cu -o bin/Polygon.obj

bin/Octree.obj: lib/hittable/Octree.cu lib/hittable/headers/Octree.hpp lib/hittable/headers/Scene.hpp lib/hittable/headers/Hitable.hpp lib/hittable/headers/HitRecord.hpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Octree.cu -o bin/Octree.obj



clean:
	rm -rf bin/ run

rebuild: clean all

clean-win:
	del /s /q bin\ run.exe	

rebuild-win: clean-win all


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