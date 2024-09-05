BIGEN = nvcc --relocatable-device-code=true

all: run.exe

# List of object files
OBJS = bin/main.o bin/vec3.o bin/ray.o bin/Sphere.o


run.exe: $(OBJS)
	$(BIGEN) -o run $(OBJS)

# Object file rules

bin/main.o: main.cu 
	$(BIGEN) -o bin/main.o -c main.cu


bin/vec3.o: lib/vec3.cu lib/headers/vec3.hpp
	$(BIGEN) -c lib/vec3.cu -o bin/vec3.o

bin/ray.o: lib/ray.cu lib/headers/ray.hpp
	$(BIGEN) -c lib/ray.cu -o bin/ray.o

bin/Sphere.o: lib/Sphere.cu lib/headers/Sphere.hpp
	$(BIGEN) -c lib/Sphere.cu -o bin/Sphere.o

run: run.exe
	./run.exe > image.ppm