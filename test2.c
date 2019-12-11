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
    double output[5][2] = { {0, 0},  {0, 0},  {0, 0},  {0, 0},  {0, 1} };
    cann_double *nnet = nnet_load("./models/nnet2");
    nnet = model_fit(nnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 1000, 0.3);
    printf("%s\n", get_ln(fopen("./yee", "r")));
    printf("%d\n", get_lines("./test_data"));
    free_nnet(nnet);
    free_data(new);
    return 0;
}
