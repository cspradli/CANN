#ifndef server_h
#define server_h

/**
 * Handles all server side stuff
 * Author: Caleb Spradlin
 * Date: 12/11/2019
 **/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/**
 * Checks for proper input
 **/
int get_input(int argc, char const *argv[]);

/**
 * Parses input for username and host
 **/
int parse_input(const char *input);

/**
 * Execs the wget through ssh
 **/
void exec_wget(char *input, int check);

void exec_job(char *input, int check);

void exec_all(char *input);

int check_input();

#endif /* server_h */