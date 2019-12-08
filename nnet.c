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
    return nnet;

}
cann_double *model_train(cann_double *nnet, int num_inputs, int num_hidden, int num_outputs, int num_training, double lr, int epochs, int training_order[], double training_in[], double training_out[]){
    double *hiddenLayer;
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
    outputWeights = nnet->output_weights;

    for (int n = 0; n < epochs; n++){

        /**
         * Need to shuffle incoming trainign order for max stochasticity
         **/
        shuffle(training_order, num_training);
        
        for (int x = 0; x < num_training; x++){
            int i = training_order[x];
            
            
            //Compute hidden layer activation
            for (int j = 0; j < num_hidden; j++){
                double activation = hiddenLayerBias[j];         // nnet->hiddenBias[j];
                for (int k = 0; k < num_inputs; k++){
                    activation = activation + 
                    training_in[i*num_training +k]*
                    nnet->hidden_weights[k*num_hidden+j];       //nnet->hidden_weights[k*num_hidden+j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            //print_array(hiddenLayer, num_hidden);
            
            
            //Comput output layer activation
            for (int j=0; j<num_outputs;j++){                   //J = num of output nodes
                double activation = outputLayerBias[j];
                for(int k=0; k<num_hidden;k++){                 // k = num of hidden nodes
                    activation += hiddenLayer[k]*
                    outputWeights[k*num_outputs + j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            
            /**
             * BAKCPROP
             * Next steps involve:
             * Calculating incremental changre in network weights
             * Moves network towards minimizing the error of the output
             * Starts at the node and works itself backward
             **/
            double deltaOut[num_outputs];
            for (int j=0; j<num_outputs; j++){
                double derivative_error = (training_out[i*num_training+j]-outputLayer[j]);
                deltaOut[j] = derivative_error * d_sigmoid(outputLayer[j]);
                if ((n%1000)==0) printf("EPOCH: %d\nDERIV OF MSE: %f\n", n,derivative_error);
            }

            /**
             * Hidden layer backprop
             * Error calculation for given node = sum of error across all output nodes
             **/
            double delta_hidden[num_hidden];
            for (int j=0; j<num_hidden; j++){
                double deriv_error = 0.0f;
                for (int k=0; k<num_outputs; k++){
                    deriv_error += deltaOut[k] * outputWeights[j*num_outputs+k];
                }
                delta_hidden[j] = deriv_error*d_sigmoid(hiddenLayer[j]);
            }

            /**
             * Apply deltas to respective weight matrices
             * in addition to bias units
             * 1st: Apply change in output weights
             * 2nd: Apply change in hidden weights
             **/
            for (int j=0; j<num_outputs;j++){
                outputLayerBias[j] += deltaOut[j]*lr;
                for(int k=0; k<num_hidden; k++){
                    outputWeights[k*num_outputs+j] += hiddenLayer[k]*deltaOut[j]*lr;
                }
            }
            for (int j=0; j<num_hidden; j++){
                hiddenLayerBias[j] += delta_hidden[j]*lr;
                for (int k=0; k<num_inputs; k++){
                    hiddenWeights[k*num_hidden+j] += training_in[i*num_training+k]*delta_hidden[j]*lr;
                }
            }


        }
    }
    print_array(hiddenLayer, num_hidden);
    print_mat(hiddenWeights, num_inputs, num_hidden);
    nnet->hidden = hiddenLayer;
    nnet->hidden_weights = hiddenWeights;
    nnet->hiddenBias = hiddenLayerBias;
    nnet->output = outputLayer;
    nnet->output_weights = outputWeights;
    nnet->outBias = outputLayerBias;
    return nnet;

}


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
    return 1 / (1 + exp(-x));
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

void print_all(cann_double *nnet){
    printf("\n\n########## MODEL SUMMARY ##########\n");
    printf("NUM INPUT: %d        NUM HIDDEN: %d\nNUM OUTPUT: %d", nnet->num_inputs, nnet->num_hidden, nnet->num_outputs);
    printf("----------VECTORS----------\nHidden:\n");
    print_array(nnet->hidden, 2);
    printf("Output\n");
    print_array(nnet->output, nnet->num_outputs);
    printf("----------WEIGHTS----------:\n");
    printf("HIDDEN: \n");
    print_mat(nnet->hidden_weights, nnet->num_inputs, nnet->num_hidden);
    printf("OUTPUT: \n");
    print_mat(nnet->output_weights, nnet->num_hidden, nnet->num_outputs);
    printf("----------BIASES-----------\n");
    printf("HIDDEN BIAS: \n");
    print_array(nnet->hiddenBias, nnet->num_hidden);
    printf("FINAL OUTPUT BIAS: \n");
    print_array(nnet->output_weights, nnet->num_outputs);
}

