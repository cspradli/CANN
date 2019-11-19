#include <stdio.h>
#include "nnet.h"


int main(int argc, char const *argv[])
{
    double tw = 2.0;
    printf("sigmoid %f\n", sigmoid(tw));
    
    test_vector(atoi(argv[1]));
    test_matrix(atoi(argv[2]), atoi(argv[3]));
    
    gsl_vector *my_vect;
    my_vect = init_vector(5);
    print_vector(my_vect);
    gsl_matrix *my_mat;
    my_mat = init_matrix(5,5);
    print_matrix(my_mat);
    gsl_vector_free(my_vect);
    gsl_matrix_free(my_mat);
    return 0;

}
