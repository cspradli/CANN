#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
#include "server.h"
int main(int argc, char const *argv[])
{
    if (argc == 3){
        printf("Running with server\n");
        get_input(argc, argv);
    } else if (argc == 2 && (!strcmp(argv[1], "n"))) {

    printf("Running locally\n");
    

    //printf("check : %d", check);
    int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;
    //int trainingorder[] = {0,1,2,3};
    double input[5][3] = { {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double output[5][2] = { {0, 0},  {0, 0},  {0, 1},  {0, 1},  {0, 0} };
    //double training_outputs[4] = { 0.0f,1.0f,1.0f,0.0f};
    //double training_inputs[8] = { 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f };

    cann_double *nnet;
    nnet = init_model_double(numInputs, numHidden, numOutputs);
    print_all(nnet);

    //nnet = model_train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 0.1, 10000, trainingorder, training_inputs, training_outputs);
    //print_all(nnet);
    nnet = model_fit_update(nnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 1000, 0.03);
    nnet = model_fit(nnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 10000, 0.03);
    //printf("Trying prediction\n");
    //double test_set[2] = { 0.0f, 0.0f };
    //forward_prop(nnet, test_set);
    //backprop(nnet, training_inputs, training_outputs, 0.1);
    //nnet = train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 0.03, 10000, trainingorder, training_inputs, training_outputs);
    //print_all(nnet);
    //double test_set[2] = { 1.00, 0.00 };
    //forward_prop(nnet, test_set);
    //print_array(predict(nnet, test_set), nnet->num_outputs);
    free_nnet(nnet);
    } else {
        printf("Usage: ./my-nnet y(y/n) (if yes)username@host.edu \n(y/n for running on server)\n(username@host.edu to use as server to run on)\n");
    }
    
    return 0;
    
}
