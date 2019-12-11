#ifndef nnet_io_h
#define nnet_io_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nnet.h"
#include <sys/socket.h>
#include <unistd.h>

/**
 * Struct to use as the incoming data
 **/
typedef struct
{
    double** target_in;
    double** target;
    int num_input;
    int num_output;
    int num_rows;
} data;

/**
 * Forks another process to create a new file to save to
 * Uses "touch" program
 **/
void check_paths(char *path);

/**
 * Mallocs 2d arrays
 **/
double **init_2d(int rows, int columns);

/**
 * Primary function for getting data from file line by line into data struct
 **/
data *get_data(char *path, int num_inputs, int num_outputs);


/**
 * Gets one line at a time from specified file destination
 **/
char *get_ln(FILE* file);

/**
 * Parses the given line to doubles to be put in an array
 **/
void parse_data(data *in, char *line, int row);

/**
 * Gets the number of lines in a file
 **/
int get_lines(char *path);

/**
 * Frees the Data struct object
 **/
void free_data(data *in);

/**
 * Function to save all weights and biases to a file
 **/
void nnet_save(cann_double *nnet, char *path);

/**
 * Loads a nnet from a file and initializes the nnet
 **/
cann_double *nnet_load(char *path);

/**
 * Forgot what this was
 **/
double *twoDoneD(double **in, int sizeX, int sizeY);


#endif /* server_h */