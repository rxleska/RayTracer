BIGEN = g++ -Wall -Werror -ggdb

all: run

# List of object files
OBJS = bin/main.o bin/camera.o bin/color.o bin/hittable_list.o bin/hittable.o bin/material.o bin/ray.o bin/sphere.o bin/tppm.o bin/vec3.o bin/Polygon.o

# Final executable
run: $(OBJS)
	$(BIGEN) -o run $(OBJS)

# Object file rules
bin/main.o: main.cpp lib/headers/interval.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -o bin/main.o -c main.cpp

bin/camera.o: lib/camera.cpp lib/headers/camera.hpp lib/headers/interval.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/camera.cpp -o bin/camera.o

bin/color.o: lib/color.cpp lib/headers/color.hpp
	$(BIGEN) -c lib/color.cpp -o bin/color.o

bin/hittable_list.o: lib/hittable_list.cpp lib/headers/hittable_list.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/hittable_list.cpp -o bin/hittable_list.o

bin/hittable.o: lib/hittable.cpp lib/headers/hittable.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/hittable.cpp -o bin/hittable.o

bin/material.o: lib/material.cpp lib/headers/material.hpp
	$(BIGEN) -c lib/material.cpp -o bin/material.o

bin/ray.o: lib/ray.cpp lib/headers/ray.hpp
	$(BIGEN) -c lib/ray.cpp -o bin/ray.o

bin/sphere.o: lib/Sphere.cpp lib/headers/Sphere.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/Sphere.cpp -o bin/sphere.o

bin/tppm.o: lib/tppm.cpp lib/headers/tppm.hpp
	$(BIGEN) -c lib/tppm.cpp -o bin/tppm.o

bin/vec3.o: lib/vec3.cpp lib/headers/vec3.hpp lib/headers/RTWeekend.hpp
	$(BIGEN) -c lib/vec3.cpp -o bin/vec3.o

bin/Polygon.o: lib/Polygon.cpp lib/headers/Polygon.hpp
	$(BIGEN) -c lib/Polygon.cpp -o bin/Polygon.o

clean:
	rm -f run bin/*.o

clean-win:
	del run.exe bin\*.o	

rebuild: clean all

