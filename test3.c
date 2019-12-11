#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
#include "server.h"
#include "nnet_io.h"

int main(int argc, char const *argv[])
{
    int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;
    data *new;
    new = get_data("./test_data", 2, 1);
    double input[5][3] = { {0,0,0}, {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double output[5][2] = { {0, 0},  {0, 0},  {1, 1},  {1, 1},  {0, 0} };
   
///*
    cann_double *mynnet = nnet_load("./models/nnet2");
    mynnet = model_fit(mynnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 1000, 0.3);
    print_all(mynnet);
    free_nnet(mynnet);
//*/
    free_data(new);
    return 0;
}
