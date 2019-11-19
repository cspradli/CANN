/**
 * NNET portion of ANN - C
 * Author Caleb Spradlin
 **/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "nnet.h"






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

