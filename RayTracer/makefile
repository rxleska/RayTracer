# compiler setup 
CC = g++

# compiler flags:
# -ggdb adds debugging information to the executable file
# -Wall turns on most, but not all, compiler warnings
# -Werror makes all warnings into errors
CFLAGS = -ggdb -Wall -Werror


PROCESSING = bin/Ray.o bin/Vec3.o
MATERIALS  =
HITTABLES  = bin/Sphere.o 

all: run 

run: bin/ bin/main.o $(PROCESSING) $(MATERIALS) $(HITTABLES)
	$(CC) $(CFLAGS) -o run main.o $(PROCESSING) $(MATERIALS) $(HITTABLES)

bin/:
	mkdir bin

bin/main.o: main.cpp 
	$(CC) $(CFLAGS) -c main.cpp -o bin/main.o

# Processing Objects

bin/Ray.o: lib/processing/Ray.cpp lib/processing/headers/Ray.hpp
	$(CC) $(CFLAGS) -c lib/processing/Ray.cpp -o bin/Ray.o

bin/Vec3.o: lib/processing/Vec3.cpp lib/processing/headers/Vec3.hpp
	$(CC) $(CFLAGS) -c lib/processing/Vec3.cpp -o bin/Vec3.o


# Material Objects


# Hittable Objects

bin/Sphere.o: lib/hittable/Sphere.cpp lib/hittable/headers/Sphere.hpp lib/processing/headers/Ray.hpp lib/hittable/headers/Hittable.hpp lib/hittable/headers/HitRecord.hpp
	$(CC) $(CFLAGS) -c lib/hittable/Sphere.cpp -o bin/Sphere.o
