/**
 * NNET portion of ANN - C
 * Author Caleb Spradlin
 **/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "nnet.h"

cann_double *init_model_double(int num_inputs, int num_hidden, int num_outputs){


    cann_double* nnet = (cann_double* )malloc(sizeof(cann_double));
    if (!nnet) return 0;

    nnet->num_inputs = num_inputs;
    nnet->num_hidden = num_hidden;
    nnet->num_inputs = num_inputs;
    nnet->num_hidden = num_hidden;
    nnet->num_outputs = num_outputs;
    /*
    nnet->hidden = (double *) malloc(num_hidden * sizeof(double));
    nnet->output = (double *) malloc(num_outputs * sizeof(double));
    nnet->hiddenBias = (double *) malloc(num_hidden*sizeof(double));
    nnet->outBias = (double *) malloc(num_outputs * sizeof(double));
    nnet->hidden_weights = (double *) malloc((num_inputs+num_hidden) * sizeof(double));
    nnet->output_weights = (double *) malloc((num_hidden+num_outputs) * sizeof(double));
    */
    nnet->hidden = (double *) calloc(num_hidden, sizeof(double));
    nnet->output = (double *) calloc(num_outputs, sizeof(double));
    nnet->hiddenBias = (double *) calloc(num_hidden, sizeof(double));
    nnet->outBias = (double *) calloc(num_outputs,  sizeof(double));
    nnet->hidden_weights = (double *) calloc((num_inputs+num_hidden), sizeof(double));
    nnet->output_weights = (double *) calloc((num_hidden+num_outputs), sizeof(double));

    nnet->hidden = init_zero(nnet->hidden, num_hidden);
    nnet->output = init_zero(nnet->output, num_outputs);
    nnet->hiddenBias = init_random(nnet->hiddenBias, num_hidden);
    nnet->outBias = init_random(nnet->outBias, num_outputs);
    nnet->hidden_weights = init_random(nnet->hidden_weights, (num_inputs+num_hidden));
    nnet->output_weights = init_random(nnet->output_weights, (num_hidden+num_outputs));
    nnet->hidden[0] = 1.0;
    printf("Vector Hidden: \n");
    print_array(nnet->hidden, num_hidden);
    printf("Vector out\n");
    print_array(nnet->output, num_outputs);
    printf("Weights:\n");
    printf("HIDDEN: \n");
    print_mat(nnet->hidden_weights, num_inputs, num_hidden);
    printf("OUTPUT: \n");
    print_mat(nnet->output_weights, num_hidden, num_outputs);
    printf("HIDDEN BIAS: \n");
    print_array(nnet->hiddenBias, num_hidden);
    printf("Final output bias: \n");
    print_array(nnet->outBias, num_outputs);
    return nnet;
}
cann_double *model_train(cann_double *nnet, int num_inputs, int num_hidden, int num_outputs, int num_training, int numhidden_layers, int epochs, int training_order[], double training_in[], double training_out[]){
    /*double *hiddenLayer;
    double *outputLayer;
    double *hiddenLayerBias;
    double *outputLayerBias;
    double *hiddenWeights;
    double *outputWeights;
    hiddenLayer = nnet->hidden;
    outputLayer = nnet->output;
    hiddenLayerBias = nnet->hiddenBias;
    outputLayerBias = nnet->outBias;
    hiddenWeights = nnet->hidden_weights;
    outputWeights = nnet->output_weights;*/
    printf("[");
    for (int i = 0; i < num_training; i++){
        printf("%d, ", training_order[i]);
    }
    printf("]\n");
    //int n = 0;
    //int x = 0;
    //int j = 0;
    //int k = 0;
    //int i = 0;
    //double activation = (double) 0;
    printf("here 1\n");
    /*
    for (n=0; n < epochs; n++){
        printf("here 2 %d\n", n);
        shuffle(trainingOrder, num_training);
        printf("here 3 %d\n", n);
            printf("[");
        for (int i = 0; i < num_training; i++){
        printf("%d, ", trainingOrder[i]);
        }
        printf("]\n");
        for (x=0; x < num_training; x++){
            printf("here 4 %d\n", x);
            i = trainingOrder[x];
            printf("here 5 %d\n", x);
            for (j=0; j <num_hidden; j++){
                printf("here 6 %d\n", j);
                activation = hiddenLayerBias[j];
                 printf("here 7 %d\n", j);
                for (k=0; k < num_inputs; k++){
                    printf("here 8 %d\n", k);
                    activation += training_in[i][k]; // hiddenWeights[k*num_hidden + j];
                    printf("here 9 %d\n", k);
                }
                printf("here 10 %d\n", j);
                hiddenLayer[j] = sigmoid(activation);
                printf("here 11 %d\n", j);
            }
        }
    }*/
    nnet->hidden_weights = init_random(nnet->hidden_weights, nnet->num_hidden);
    nnet->hiddenBias = init_random(nnet->hiddenBias, nnet->num_hidden);

    for (int n = 0; n < epochs; n++){
        printf("EPOCH: %d\n", n);
        shuffle(training_order, num_training);
        for (int x = 0; x < num_training; x++){
            int i = training_order[x];
            //Compute hidden layer activation
            for (int j = 0; j < num_hidden; j++){
                printf("here 01 %d,%d,%d\n", n, x, j);
                printf("%f\n", nnet->hiddenBias[j]);
                double activation = nnet->hiddenBias[j];
                printf("here 02 %d,%d,%d\n", n, x, j);
                for (int k = 0; k < num_inputs; k++){
                    printf("here 8 %d,%d,%d,%d\n", n, x, j, k);
                    activation = activation + training_in[i*num_training +k]* nnet->hidden_weights[k*num_hidden+j];// gsl_matrix_get(hiddenWeights, k, j);
                    printf("here 9 %d,%d,%d,%d\n", n, x, j, k);
                }
                printf("here 10 %d,%d,%d\n", n, x, j);
                            printf("[");
                for (int i = 0; i < num_training; i++){
                printf("%f, ", nnet->hidden[i]);
                }
                printf("] %d, %f\n", j,nnet->hidden[j]);
                nnet->hidden[j] = sigmoid(activation);
                printf("here 11 %d,%d,%d\n", n, x, j);
                //gsl_vector_set(hiddenLayer, j, sigmoid(activation));
            }
            //print_array(hiddenLayer, num_hidden);
            //Comput output layer activation
        }
    }
    return nnet;

}

/*

    nnet->hidden = (double *) calloc(num_hidden, sizeof(double));
    nnet->output = (double *) calloc(num_outputs, sizeof(double));
    nnet->hiddenBias = (double *) calloc(num_hidden, sizeof(double));
    nnet->outBias = (double *) calloc(num_outputs,  sizeof(double));
    nnet->hidden_weights = (double *) calloc((num_inputs+num_hidden), sizeof(double));
    nnet->output_weights = (double *) calloc((num_hidden+num_outputs), sizeof(double));
*/
/* ************** HELPER FUNCTIONS ***************** */
void free_nnet(cann_double *in){
    free(in->output_weights);
    free(in->hidden_weights);
    free(in->outBias);
    free(in->hiddenBias);
    free(in->output);
    free(in->hidden);
    free(in);
}

double sigmoid(double x){
    double out = 0;
    out = 1 / (1 + exp(-x));
    printf("%f\n", out);
    return out;
    }

double d_sigmoid(double x){
    return x * (1 - x);
}

double tanh_(double x){
    return 2 / 1 + exp(-2*x) - 1;
}

double d_tanh(double x){
    return 1 - pow((2/1+exp(-2*x) -1), 2);
}

double init_weights(){
    return ((double) rand())/((double) RAND_MAX);
}


void print_array(double input[], int length){
    printf("\n");
    for (int i=0; i < length; i++){
        printf("|%f|\n", input[i]);
    }
    printf("\n");
}

void print_mat(double input[], int lengthX, int lengthY){
    
    for (int x=0; x < lengthX; x++){
        printf("| ");
        for (int y=0; y < lengthY; y++){
            printf("%f ", input[x * lengthX + y]);
        }
        printf("|\n");
    }
    printf("\n");
}


double *init_zero(double input[], int length){
    for (int x=0; x < length; x++){
        input[x] = (double) 0;
    }
    return input;
}

double *init_random(double input[], int length){
    for (int x=0; x < length; x++){
        input[x] = init_weights();
    }
    return input;
}

void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle(int arr[], int n){
    srand( time(NULL) );
    for (int i = n-1; i > 0; i--){
        int j = rand() % (i+1);
        swap(&arr[i], &arr[j]);
    }
}

