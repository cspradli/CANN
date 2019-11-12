#
# Makefile for ANN - C
#


CC = gcc
CFLAGS == -O -Wall
LFLAGS = 

all: nnet

nnet: main.c nnet.c nnet.h activation.c activation.h
	$(CC) $(CFLAGS) -c -o nnet.o nnet.c
	$(CC) $(CFLAGS) -c -o main.o main.c
	$(CC) $(CFLAGS) -c -o activation.c activation.h
	$(CC) $(LFLAGS) -O nnet main.o nnet.o activation.o

clean:
	rm -f activation.o nnet.o main.o nnet