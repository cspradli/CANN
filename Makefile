#
# Makefile for ANN - C
#


CC = gcc
CFLAGS = -O -Wall
LFLAGS = -lm

all: my-nnet

my-nnet: main.c nnet.c nnet.h server.c server.h
	$(CC) $(CFLAGS) -c -o nnet.o nnet.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o server.o server.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o main.o main.c $(LFLAGS)
	$(CC) $(LFLAGS) -o my-nnet main.o nnet.o server.o $(LFLAGS)

clean:
	rm -f server.o nnet.o main.o my-nnet