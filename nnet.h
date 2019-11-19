#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>


/**
 * Takes a vector and returns same matrix with random values between 0.0 and 1.0
 **/
void vectorInit_random(gsl_vector *my_vect);

/**
 * Takes a vector and returns same vector with random values between 0.0 and 1.0
 **/
void matrixInit_random(gsl_matrix *my_mat);

/**
 * Init all weights and biases between 0.0 and 1.0
 **/
double init_weights();

/**
 * Initializes a vector to size 'size'
 * Returns a gsl_vector with all values set to zero
 **/
gsl_vector* init_vector(int size);

/**
 * Inits a matric to size(sizeX, sizeY)
 * Returns a matrix of size with all values set to zero
 **/
gsl_matrix* init_matrix(int sizeX, int sizeY);

/**
 * Prints a vector in formatted style
 **/
void print_vector(gsl_vector *in);

/**
 * Prints a matrix in formatted style
 **/
void print_matrix(gsl_matrix *in);

/**
 * Sigmoid activation function
 * Function which is plotted as 'S' shaped graph
 * Nature: Non-Linear
 * Value Range: 0 to 1
 * Uses: Used in output layer of a binary classification
 **/
double sigmoid(double x);

/**
 * Derivative of sigmoid function
 **/
double d_sigmoid(double x);

/**
 * Tangent Hyperbolic function
 * Mathematically shifted version of the sigmoid function
 * Nature: Non-Linear
 * Value: -1 to +1
 * Uses: Usually in hideen layers of a nnet
 **/
double tanh_(double x);

/**
 * Derivative of tanh function
 **/
double d_tanh(double x);

