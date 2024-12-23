# Compiler Setup
CC = nvcc 

# Compiler Flags
CFLAGS = -arch=sm_89 -rdc=true 


all: run 

run: bin bin/main.obj 
	$(CC) $(CFLAGS) -o run bin/main.obj

bin:
	mkdir bin


bin/main.obj: main.cu
	$(CC) $(CFLAGS) -c main.cu -o bin/main.obj