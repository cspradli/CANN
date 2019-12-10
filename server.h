#ifndef server_h
#define server_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int get_input(int argc, char const *argv[]);

int parse_input(const char *input);

void exec_wget(char *input, int check);

void exec_job(char *input, int check);

void exec_all(char *input);

int check_input();

#endif /* server_h */