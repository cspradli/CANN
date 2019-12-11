#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
#include "server.h"
#include "nnet_io.h"
int main(int argc, char const *argv[])
{   
    pid_t pid;
    if (argc == 3){
        //Spin off fork to run job on server, wait to hear
        if ((pid = fork() == 0)){
        printf("Running with server\n");
        get_input(argc, argv);
        }
        wait(NULL);
    } else if (argc == 2 && (!strcmp(argv[1], "n"))) {

    printf("Running locally\n");
    

    int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;
    double input[5][3] = { {0,0,0}, {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double output[5][2] = { {0, 0},  {0, 0},  {0, 0},  {0, 0},  {0, 1} };

    cann_double *nnet;
    nnet = init_model_double(numTrainingSets,numInputs, numHidden, numOutputs);
    print_all(nnet);

    nnet = model_fit(nnet, numTrainingSets, numInputs, numHidden, numOutputs, input, output, 10000, 0.03);
    print_all(nnet);
    nnet_save(nnet, "./yee");
    free_nnet(nnet);
    cann_double *my_net;
    my_net = nnet_load("./yee");
    print_all(my_net);
    free_nnet(my_net);
    } else {
        printf("Usage: ./my-nnet y(y/n) (if yes)username@host.edu \n(y/n for running on server)\n(username@host.edu to use as server to run on)\n");
    }
    
    return 0;
    
}
