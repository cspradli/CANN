#
# Makefile 
#

CC = gcc
CFLAGS = -O -Wall 
LFLAGS = -pthread

all: multi_curl

multi_curl: main.c linked_list.c linked_list.h mycurl.c mycurl.h
	$(CC) $(CFLAGS) -c -o linked_list.o linked_list.c
	$(CC) $(CFLAGS) -c -o mycurl.o mycurl.c
	$(CC) $(CFLAGS) -c -o main.o main.c
	$(CC) $(LFLAGS) -o multi_curl main.o linked_list.o mycurl.o

clean:
	rm -f linked_list.o mycurl.o main.o multi_curl

test: multi_curl
	./multi_curl
