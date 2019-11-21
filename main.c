#include <stdio.h>
#include "nnet.h"


int main(int argc, char const *argv[])
{
    double tw = 2.0;
    printf("sigmoid %f\n", sigmoid(tw));
    
    gsl_vector *my_vect;
    my_vect = init_vector(5);
    print_vector(my_vect);
    gsl_matrix *my_mat;
    my_mat = init_matrix(5,5);
    print_matrix(my_mat);
    vectorInit_random(my_vect);
    print_vector(my_vect);
    printf("%f\n", gsl_vector_get(my_vect, 0));
    matrixInit_random(my_mat);
    print_matrix(my_mat);

    gsl_matrix_transpose(my_mat);
    printf("Transpose Vector\n");
    print_matrix(my_mat);
    gsl_vector_free(my_vect);
    gsl_matrix_free(my_mat);
    int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int trainingorder[] = {0,1,2,3};
    double training_outputs[4][1] = { {0.0f},{1.0f},{1.0f},{0.0f} };
    double training_inputs[4][2] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
    init_model(numInputs, 2, numOutputs, numTrainingSets, 1, 100000, trainingorder, training_inputs, training_outputs);
    return 0;

}
