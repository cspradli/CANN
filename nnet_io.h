#ifndef nnet_io_h
#define nnet_io_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nnet.h"

typedef struct
{
    double** target_in;
    double** target;
    int num_input;
    int num_output;
    int num_rows
} data;

double **init_2d(int rows, int columns);

data *get_data(int num_inputs, int num_outputs, int num_rows);

void free_data(data *in);

void nnet_save(cann_double *nnet, char *path);

cann_double *nnet_load(char *path);



#endif /* server_h */