#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>

void test_vector(int in);

gsl_matrix *matrixInit_random(int x, int y);

double init_weights();

gsl_vector* init_vector(int size);

gsl_matrix* init_matrix(int sizeX, int sizeY);

void print_vector(gsl_vector *in);

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

