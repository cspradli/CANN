/**
 * NNET header file
 * Date: 12/08/19
 * Author: Caleb Spradlin
 **/


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

/**
 * Initializes a neural net
 **/
cann_double *init_model_double(int num_inputs, int num_hidden, int num_outputs);

/**
 * Performs the training routine 
 **/
cann_double *model_train(cann_double *nnet, int num_inputs, int num_hidden, int num_outputs, int num_training, int numhidden_layers, int epochs, int trainingOrder[], double training_in[], double training_out[]);

/**
 * Takes a vector and returns same vector with random values between 0.0 and 1.0
 **/
double *init_random(double input[], int length);

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


/**
 * Print all
 **/
void print_all(cann_double *in);