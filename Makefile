main: main.c matrix.o
	gcc main.c matrix.o -ggdb -o main -lm

matrix.o: matrix.h matrix.c
	gcc matrix.c -c -ggdb -o matrix.o
