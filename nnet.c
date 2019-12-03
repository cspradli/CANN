/**
 * NNET portion of ANN - C
 * Author Caleb Spradlin
 **/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "nnet.h"


void init_model_vectorized(int num_inputs, int num_hidden,int num_outputs, int num_training, int num_hiddenlayers, int epochs, int training_order[], double training_in[][num_inputs],double training_out[][num_outputs]){
    gsl_vector *hidden;
    gsl_vector *output;
    gsl_vector *hiddenBias;
    gsl_vector *outBias;
    gsl_vector *deltaHidden;
    gsl_vector *deltaOut;
    gsl_matrix *hiddenWeights;
    gsl_matrix *outputWeights;
    hidden = init_vector(num_hidden);
    output = init_vector(num_outputs);
    hiddenBias = init_vector(num_hidden);
    outBias = init_vector(num_outputs);
    deltaHidden = init_vector(num_hidden);
    deltaOut = init_vector(num_outputs);
    hiddenWeights = init_matrix(num_inputs, num_hidden);
    matrixInit_random(hiddenWeights);
    outputWeights = init_matrix(num_hidden, num_outputs);
    matrixInit_random(outputWeights);
    double lr = 0.1f;

    //hidden = gsl_matrix_transpose();
}

void init_model(int num_inputs, int num_hidden, int num_outputs, int num_training, int num_hiddenlayers, int epochs, int training_order[], double training_in[][num_inputs], double training_out[][num_outputs]){
    gsl_vector *hidden;
    gsl_vector *output;
    gsl_vector *hiddenBias;
    gsl_vector *outBias;
    gsl_vector *deltaHidden;
    gsl_vector *deltaOut;
    gsl_matrix *hiddenWeights;
    gsl_matrix *outputWeights;
    hidden = init_vector(num_hidden);
    output = init_vector(num_outputs);
    hiddenBias = init_vector(num_hidden);
    outBias = init_vector(num_outputs);
    deltaHidden = init_vector(num_hidden);
    deltaOut = init_vector(num_outputs);
    hiddenWeights = init_matrix(num_inputs, num_hidden);
    matrixInit_random(hiddenWeights);
    outputWeights = init_matrix(num_hidden, num_outputs);
    matrixInit_random(outputWeights);
    double lr = 0.1f;

    for (int n = 0; n < epochs; n++){
        shuffle(training_order, num_training);
        for (int x = 0; x < num_training; x++){
            int i = training_order[x];
            //Compute hidden layer activation
            for (int j = 0; j < num_hidden; j++){
                double activation = gsl_vector_get(hiddenBias, j);
                for (int k = 0; k < num_inputs; k++){
                    activation += training_in[i][k]*gsl_matrix_get(hiddenWeights, k, j);
                }
                gsl_vector_set(hidden, j, sigmoid(activation));
            }
            //print_vector(hidden);
            //Comput output layer activation
            for (int j = 0; j < num_outputs; j++){
                double activation = gsl_vector_get(outBias, j);
                
                for (int k = 0; k < num_hidden; k++){
                    activation += (gsl_vector_get(hidden, k) * gsl_matrix_get(outputWeights, k, j));
                }
                gsl_vector_set(output, j, sigmoid(activation));
                
            }
            
            //printf("Input %f %f Output: %f\n", training_in[i][0], training_in[i][1], gsl_vector_get(output, i));
            //Compute change in output weights
            for (int j = 0; j < num_outputs; j++){
                double dError = (training_out[i][j]-gsl_vector_get(output, j));
                gsl_vector_set(deltaOut, j, (dError * d_sigmoid(gsl_vector_get(output, j))));
            }
            
            //Compute change in hidden weights
            for (int j = 0; j < num_hidden; j++){
                double dError = 0.0f;
                for (int k = 0; k < num_outputs; k++){
                    dError += (gsl_vector_get(deltaOut, k) * gsl_matrix_get(outputWeights, j, k));
                }
                gsl_vector_set(deltaHidden, j, (dError * d_sigmoid(gsl_vector_get(hidden, j))));
            }
            
            for (int j = 0; j < num_outputs; j++){
                double tempa = gsl_vector_get(outBias, j);
                gsl_vector_set(outBias, j, (tempa + (gsl_vector_get(deltaOut, j) * lr)));
                for (int k =0; k < num_hidden; k++){
                    double tempb = gsl_matrix_get(outputWeights, k, j);
                    gsl_matrix_set(outputWeights, k, j, (tempb + (gsl_vector_get(hidden, k)*gsl_vector_get(deltaOut, j)*lr)));
                }
            }
            for (int j = 0; j < num_hidden; j++){
                double tempa = gsl_vector_get(hiddenBias, j);
                gsl_vector_set(hiddenBias, j, (tempa + (gsl_vector_get(deltaHidden, j) * lr)));
                for (int k =0; k < num_hidden; k++){
                    double tempb = gsl_matrix_get(hiddenWeights, k, j);
                    gsl_matrix_set(hiddenWeights, k, j, (tempb + (training_in[i][k]*gsl_vector_get(deltaHidden, j)*lr)));
                }
            }
            
        }
    }

    printf("Vector Hidden: \n");
    print_vector(hidden);
    printf("Vector out\n");
    print_vector(output);
    printf("Weights:\n");
    printf("HIDDEN: \n");
    print_matrix(hiddenWeights);
    printf("OUTPUT: \n");
    print_matrix(outputWeights);
    printf("HIDDEN BIAS: \n");
    print_vector(hiddenBias);
    printf("Final output bias: \n");
    print_vector(outBias);
    gsl_vector_free(hidden);
    gsl_vector_free(output);
    gsl_vector_free(hiddenBias);
    gsl_vector_free(outBias);
    gsl_vector_free(deltaHidden);
    gsl_vector_free(deltaOut);
    gsl_matrix_free(hiddenWeights);
    gsl_matrix_free(outputWeights);

}



/* ************** HELPER FUNCTIONS ***************** */
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

gsl_vector* init_vector(int size){
    gsl_vector *vect = gsl_vector_alloc(size);
    gsl_vector_set_all(vect, 0.0);
    return vect;
}

gsl_matrix* init_matrix(int sizeX, int sizeY){
    gsl_matrix *mat = gsl_matrix_alloc(sizeX, sizeY);
    gsl_matrix_set_all(mat, 0.0);
    return mat;
}

void print_vector(gsl_vector *in){
    printf("\n");
    for (int y = 0; y < in->size; y++){
        double ret = gsl_vector_get(in, y);
        printf("|%f|\n", ret);
    }
    printf("\n");
}

void print_matrix(gsl_matrix *in){
    printf("\n");
    for (int a = 0; a < in->size1; a++){
        printf("|");
        for (int b = 0; b < in->size2; b++){
            double ret = gsl_matrix_get(in, a, b);
            printf(" %f ", ret);
        }
        printf("|\n");
    }
    printf("\n");
}

void vectorInit_random(gsl_vector *my_vect){
    for (int i = 0; i < my_vect->size; i++){
        gsl_vector_set(my_vect, i, init_weights());   
    }
    //gsl_vector_set_all(my_vect, init_weights());
    return;
}

void matrixInit_random(gsl_matrix *my_mat){
    for (int i = 0; i < my_mat->size1; i++){
        for (int j = 0; j < my_mat->size2; j++){
            gsl_matrix_set(my_mat, i, j, init_weights());
        }
    }
    return;
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

