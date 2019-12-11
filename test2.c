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
/*
    cann_double *nnet;
    nnet = init_model_double(numTrainingSets,numInputs, numHidden, numOutputs, 0);
    print_all(nnet);

    nnet = model_fit(nnet,numTrainingSets, numInputs, numHidden, numOutputs, input, output, 10000, 0.03);
    print_all(nnet);
    nnet_save(nnet, "./models/nnet2");
    free_nnet(nnet);
*/    
///*
    cann_double *mynnet = nnet_load("./models/nnet2");
    mynnet = model_fit(mynnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 1000, 0.3);
    nnet_save(mynnet, "./models/mynnet");
    print_all(mynnet);
    free_nnet(mynnet);
//*/
    free_data(new);
    return 0;
}
