#include <stdio.h>
#include "nnet.h"


int main(int argc, char const *argv[])
{
    /*double tw = 2.0;
    printf("sigmoid %f\n", sigmoid(tw));
    
    gsl_matrix *my_mat2;
    gsl_matrix *my_mat;
    my_mat = init_matrix(5,5);
    print_matrix(my_mat);
    my_mat2 = init_matrix(5, 1);
    matrixInit_random(my_mat2);
    print_matrix(my_mat2);
    printf("%f\n", gsl_matrix_get(my_mat2, 1, 0));
    matrixInit_random(my_mat);
    print_matrix(my_mat);
    gsl_matrix_transpose(my_mat);
    printf("Transpose Vector\n");
    print_matrix(my_mat);
    gsl_matrix_mul_elements(my_mat2, my_mat);
    
    gsl_matrix_free(my_mat2);
    gsl_matrix_free(my_mat);*/
    //int numTrainingSets = 4;
    int numInputs = 2;
    int numOutputs = 1;
    int numHidden = 2;
    //int trainingorder[] = {0,1,2,3};
    //double training_outputs[4][1] = { {0.0f},{1.0f},{1.0f},{0.0f} };
    //double training_inputs[4][2] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
    //cann *my_nnet;
    //my_nnet = init_model(numInputs, 2, numOutputs, numTrainingSets, 1, 100000, trainingorder, training_inputs, training_outputs);
    //my_nnet = train_model(my_nnet, 2, numInputs, numOutputs, numTrainingSets, 1000, trainingorder, training_inputs, training_outputs);
    //printf("From main");
    //print_vector(my_nnet->hidden);
    //free(my_nnet);
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
    free(nnet);
    return 0;

}
