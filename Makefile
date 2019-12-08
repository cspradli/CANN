#
# Makefile for ANN - C
#


CC = gcc
CFLAGS = -O -Wall
LFLAGS = -lm

all: my-nnet

my-nnet: main.c nnet.c nnet.h
	$(CC) $(CFLAGS) -c -o nnet.o nnet.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o main.o main.c $(LFLAGS)
	$(CC) $(LFLAGS) -o my-nnet main.o nnet.o $(LFLAGS)

clean:
	rm -f nnet.o main.o my-nnet