#include <stdio.h>
#include "nnet.h"


int main(int argc, char const *argv[])
{
 
    //int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;/*
    int trainingorder[] = {0,1,2,3};
    double training_outputs[4] = { 0.0f,1.0f,1.0f,0.0f};
    double training_inputs[8] = { 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f };*/

    cann_double *nnet;
    nnet = init_model_double(numInputs, numHidden, numOutputs);
    printf("\n\n###########FROM MAIN################\n\nVector Hidden: \n");
    print_array(nnet->hidden, 2);
    printf("Vector out\n");
    print_array(nnet->output, numOutputs);
    printf("Weights:\n");
    printf("HIDDEN: \n");
    print_mat(nnet->hidden_weights, numInputs, numHidden);
    printf("OUTPUT: \n");
    print_mat(nnet->output_weights, numHidden, numOutputs);
    printf("HIDDEN BIAS: \n");
    print_array(nnet->hiddenBias, numHidden);
    printf("Final output bias: \n");
    print_array(nnet->output_weights, numOutputs);
    //nnet = model_train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 1, 1000, trainingorder, training_inputs, training_outputs);
    free_nnet(nnet);
    return 0;

}
