all: run

run: main.o
	g++ -o run main.o -lm

main: main.cpp
	g++ -o main.o main.cpp -lm

clean:
	rm -f run main.o