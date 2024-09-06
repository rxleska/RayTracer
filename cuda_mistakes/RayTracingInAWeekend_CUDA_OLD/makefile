BIGEN = nvcc --relocatable-device-code=true

all: run.exe

# List of object files
OBJS = bin/main.o bin/camera.o bin/color.o bin/hittable_list.o bin/hittable.o bin/material.o bin/ray.o bin/sphere.o bin/tppm.o bin/vec3.o bin/DynamicArray.o bin/GlobaL.o

# Final executable
run.exe: $(OBJS)
	$(BIGEN) -o run $(OBJS)

# Object file rules
bin/main.o: main.cu lib/headers/interval.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -o bin/main.o -c main.cu

bin/camera.o: lib/camera.cu lib/headers/camera.hpp lib/headers/interval.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/camera.cu -o bin/camera.o

bin/color.o: lib/color.cu lib/headers/color.hpp
	$(BIGEN) -c lib/color.cu -o bin/color.o

bin/hittable_list.o: lib/hittable_list.cu lib/headers/hittable_list.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/hittable_list.cu -o bin/hittable_list.o

bin/hittable.o: lib/hittable.cu lib/headers/hittable.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/hittable.cu -o bin/hittable.o

bin/material.o: lib/material.cu lib/headers/material.hpp
	$(BIGEN) -c lib/material.cu -o bin/material.o

bin/ray.o: lib/ray.cu lib/headers/ray.hpp
	$(BIGEN) -c lib/ray.cu -o bin/ray.o

bin/sphere.o: lib/Sphere.cu lib/headers/Sphere.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/Sphere.cu -o bin/sphere.o

bin/tppm.o: lib/tppm.cu lib/headers/tppm.hpp
	$(BIGEN) -c lib/tppm.cu -o bin/tppm.o

bin/vec3.o: lib/vec3.cu lib/headers/vec3.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/vec3.cu -o bin/vec3.o

bin/DynamicArray.o: lib/DynamicArray.cu lib/headers/DynamicArray.hpp
	$(BIGEN) -c lib/DynamicArray.cu -o bin/DynamicArray.o

bin/GlobaL.o: lib/GlobaL.cu lib/headers/GlobaL.hpp
	$(BIGEN) -c lib/GlobaL.cu -o bin/GlobaL.o

clean: # windows
	del bin\*.o
	del run.exe


rebuild: clean all


test: run.exe
	ncu -o profile run.exe
	