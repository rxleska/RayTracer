BIGEN = nvcc -diag-suppress 20012 -rdc true --debug -lineinfo

all: run.exe

# List of object files
OBJS = bin/main.o bin/camera.o bin/color.o bin/hittable_list.o bin/hittable.o bin/material.o bin/ray.o bin/sphere.o bin/tppm.o bin/vec3.o bin/DynamicArray.o bin/Global.o

# Final executable
run.exe: $(OBJS)
	$(BIGEN) --verbose -o run $(OBJS)

# Object file rules
bin/main.o: main.cu lib/headers/interval.hpp lib/headers/RTWeekend.hpp lib/headers/Global.hpp
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

bin/Global.o: lib/Global.cu lib/headers/Global.hpp
	$(BIGEN) -c lib/Global.cu -o bin/Global.o


clean:
	rm -f run bin/*.o run.exe run.exp run.lib

clean-win: 
	del run.exe bin\*.o run.exp run.lib

rebuild: clean all

# using nsight test
test: run.exe
	ncu ./run
	

san: run.exe
	compute-sanitizer --force-blocking-launches --generate-coredump ./run.exe