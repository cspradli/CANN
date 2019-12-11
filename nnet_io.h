#ifndef nnet_io_h
#define nnet_io_h

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nnet.h"


void nnet_save(cann_double *nnet, char *path);

cann_double *nnet_load(char *path);



#endif /* server_h */