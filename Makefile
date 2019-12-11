#
# Makefile for ANN - C
#


CC = gcc
CFLAGS = -O -Wall
LFLAGS = -lm -pthread

all: my-nnet

my-nnet: main.c nnet.c nnet.h server.c server.h nnet_io.c nnet_io.h
	$(CC) $(CFLAGS) -c -o nnet.o nnet.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o server.o server.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o nnet_io.o nnet_io.c $(LFLAGS)
	$(CC) $(CFLAGS) -c -o main.o main.c $(LFLAGS)
	$(CC) $(LFLAGS) -o my-nnet main.o nnet.o server.o nnet_io.o $(LFLAGS)

clean:
	rm -f nnet_io.o server.o nnet.o main.o my-nnet