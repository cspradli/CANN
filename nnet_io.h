#ifndef nnet_io_h
#define nnet_io_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nnet.h"
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>
typedef struct
{
    double** target_in;
    double** target;
    int num_input;
    int num_output;
    int num_rows;
} data;
/*
struct arg {
    data *my_data;
    char *path;
    int num_rows;
};*/

int check_load(int argc, char const *argv[]);

void *worker_thread(void *targ);

double **init_2d(int rows, int columns);

data *get_data(char *path, int num_inputs, int num_outputs);

char *get_ln(FILE* file);

void parse_data(data *in, char *line, int row);

int get_lines(char *path);

void free_data(data *in);

void nnet_save(cann_double *nnet, char *path);

cann_double *nnet_load(char *path);

double *twoDoneD(double **in, int sizeX, int sizeY);


#endif /* server_h */