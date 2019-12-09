#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
int main(int argc, char const *argv[])
{
    if (argc < 2){
        printf("USAGE: './my-nnet (send to server)y/n (sftp)y/n'\n");
    }
    if (strcmp(argv[1], "y") == 0){
        printf("SENDING JOB TO SERVER\n");
        if (fork() == 0){ //spin off child to run server commands
        printf("get command to ssh\n");
        if(system("ssh cspradli@montreat.cs.unca.edu 'wget http://arden.cs.unca.edu/~cspradli/my-nnet'"));
        if(system("ssh cspradli@montreat.cs.unca.edu './my-nnet n'")) printf("success\n");
        }
    } else {

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
    nnet = model_train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 0.1, 10000, trainingorder, training_inputs, training_outputs);
    print_all(nnet);

    printf("Trying prediction\n");
    //double test_set[2] = { 0.0f, 0.0f };
    //forward_prop(nnet, test_set);
    //backprop(nnet, training_inputs, training_outputs, 0.1);
    //nnet = train(nnet, numInputs, numHidden, numOutputs, numTrainingSets, 0.03, 10000, trainingorder, training_inputs, training_outputs);
    //print_all(nnet);
    double test_set[2] = { 1.00, 0.00 };
    forward_prop(nnet, test_set);
    print_array(predict(nnet, test_set), nnet->num_outputs);
    free_nnet(nnet);
    }
    return 0;
    
}
