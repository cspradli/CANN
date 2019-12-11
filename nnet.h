#ifndef nnet_h
#define nnet_h
/**
 * NNET header file
 * Date: 12/08/19
 * Author: Caleb Spradlin
 **/
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <pthread.h>


typedef struct cann_double{
    int num_inputs;
    int num_weights;
    int num_bias;
    int num_hidden;
    int num_outputs;
    int num_training;
    int prev_trained;
    double lr;
    double err;
    double *hidden;
    double *output;
    double *hidden_weights; 
    double *output_weights;
} cann_double;


/*
struct arg {
 //   double *a;
 //   double *b;
 //   int p;
 //   int x;
 //   int y;
   int size;
    double *a;
    double *b;
 //   int argc;
//    char **argv;
};*/



/**
 * Initializes a neural net
 **/
cann_double *init_model_double(int num_training, int num_inputs, int num_hidden, int num_outputs, int prev_trained);



cann_double *model_fit(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr);


void multi_train(int num_threads,int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr);


void *thread(void *targ);

//void *spin_thread(void *larg);
/**
 * Performs the training routine 
 **/

/**
 * Takes a vector and returns same vector with random values between 0.0 and 1.0
 **/
double *init_random(double input[], int length);

double init_further();

/**
 * Init all weights and biases between 0.0 and 1.0
 **/
double init_weights();

/**
 * Prints a vector in formatted style
 **/
void print_array(double input[], int length);

/**
 * Prints a matrix in formatted style
 **/
void print_mat(double input[], int lengthX, int lengthY);

/**
 * Takes an array or matrix and returns same one with all zeros
 **/
double *init_zero(double input[], int length);

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

/**
 * Simple swap function to swap pointer
 */
void swap(int *a, int *b);

/**
 * Shuffles the array as input
 **/
void shuffle(int arr[], int n);

/**
 * Frees all parts of a model
 **/
void free_nnet(cann_double *in);

void copy_array(double arr2[], double arr1[], int size);
/**
 * Print all
 **/
void print_all(cann_double *in);

void save_all(cann_double *in);

void read_all(FILE *in);

#endif /* server_h */

