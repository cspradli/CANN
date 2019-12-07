#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>

typedef struct cann{
    int num_inputs;
    int num_hidden;
    int num_outputs;
    int total_weights;
    int total_perceptrons;
    double lr;
    gsl_vector *hidden;
    gsl_vector *output;
    gsl_vector *hiddenBias;
    gsl_vector *outBias;
    gsl_vector *deltaHidden;
    gsl_vector *deltaOut;
    gsl_matrix *hiddenWeights;
    gsl_matrix *outputWeights;
} cann;

typedef struct twoD
{
    double **array;
    size_t rows;
    size_t cols;
}twoD;

typedef struct cann_double{
    int num_inputs;
    int num_weights;
    int num_bias;
    int num_hidden;
    int num_outputs;
    double lr;
    double *hidden;
    double *output;
    double *hiddenBias;
    double *outBias;
    double *hidden_weights; 
    double *output_weights;
} cann_double;

cann_double *init_model_double(int num_inputs, int num_hidden, int num_outputs);


cann *init_model(int num_inputs, int num_hidden, int num_outputs, int num_training, int numhidden_layers, int epochs, int trainingOrder[], double training_in[][num_inputs], double training_out[][num_outputs]);

cann *train_model(cann *input_model, int num_hidden, int num_inputs, int num_outputs, int num_training, int epochs, int training_order[], double training_in[][num_inputs], double training_out[][num_outputs]);

void makeTwoD(struct twoD *p);

void freeTwoD(struct twoD *p);

void print_array(double input[], int length);

void print_mat(double input[], int lengthX, int lengthY);
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

void swap(int *a, int *b);

void shuffle(int arr[], int n);

