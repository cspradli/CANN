#
# Makefile for ANN - C
#


CC = gcc
CFLAGS = -O -Wall
LFLAGS = 

all: my-nnet

my-nnet: main.c nnet.c nnet.h
	$(CC) $(CFLAGS) -c -o nnet.o nnet.c -lm
	$(CC) $(CFLAGS) -c -o main.o main.c -lm
	$(CC) $(LFLAGS) -o my-nnet main.o nnet.o -lm

clean:
	rm -f nnet.o main.o my-nnet