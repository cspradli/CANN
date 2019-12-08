#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
int main(int argc, char const *argv[])
{
    /*if (argc < 2){
        printf("Need more args\n");
    }
    if (strcmp(argv[1], "yes") == 0){
        printf("SENDING JOB TO SERVER\n");
        printf("get command to ssh\n");
        if(system("ssh cspradli@montreat.cs.unca.edu './my-nnet no'")) printf("success\n");
    } else {*/

    //printf("check : %d", check);
    int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;
    int trainingorder[] = {0,1,2,3};
    double training_outputs[4] = { 0.0f,1.0f,1.0f,0.0f};
    double training_inputs[8] = { 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f };

    cann_double *nnet;
    nnet = init_model_double(numInputs, numHidden, numOutputs);
    print_all(nnet);
    nnet = model_train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 1, 1000, trainingorder, training_inputs, training_outputs);
    print_all(nnet);
    free_nnet(nnet);
    //}
    return 0;
    
}
