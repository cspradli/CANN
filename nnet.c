/**
 * NNET portion of ANN - C
 * Author Caleb Spradlin
 **/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "nnet.h"






#define rando() ((double)rand()/((double)RAND_MAX+1))

cann_double *init_model_double(int num_training, int num_inputs, int num_hidden, int num_outputs){


    cann_double* nnet = (cann_double* )malloc(sizeof(cann_double));
    if (!nnet) return 0;

    nnet->num_inputs   = num_inputs;
    nnet->num_hidden   = num_hidden;
    nnet->num_inputs   = num_inputs;
    nnet->num_hidden   = num_hidden;
    nnet->num_outputs  = num_outputs;
    nnet->num_training = num_training;
    nnet->s_hidden     = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->s_out        = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->d_out        = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->s_do         = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->d_hidden     = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->dw_IH        = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->dw_HO        = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->hidden       = (double *) calloc(((num_training+1)*(num_hidden+1)), sizeof(double));
    nnet->output       = (double *) calloc(((num_training+1)*(num_outputs+1)), sizeof(double));
    nnet->hidden_weights = (double *) calloc(((num_inputs+1)*(num_hidden+1)), sizeof(double));
    nnet->output_weights = (double *) calloc(((num_hidden+1)*(num_outputs+1)), sizeof(double));

    return nnet;

}


cann_double *model_fit(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training+1];
    double s_hidden[num_training+1][num_hidden+1];
    double s_out[num_training+1][num_output+1];
    double d_out[num_output+1], s_do[num_hidden+1], d_hidden[num_hidden+1];
    double dw_IH[num_input+1][num_hidden+1], dw_HO[num_hidden+1][num_output+1];
    double hidden[((num_training+1)*(num_hidden+1))], output[((num_training+1)*(num_output+1))];
    double w_IH[((num_input+1)*(num_hidden+1))], w_HO[((num_hidden+1)*(num_output+1))];
    
    //double bias_dih[num_hidden+1], bias_who[num_output+1], bias_dho[num_output+1], bias_wih[num_hidden+1];

    double err, eta = 0.5, alpha = 0.95;
    
    int hiddenX, outputX, w_ihX, w_hoX;

    hiddenX = (num_training+1); 
    outputX = (num_training+1); 
    w_ihX = (num_input+1);
    w_hoX = (num_hidden+1);

    /**
     * Initialization of all weights and derivative of weights
     * Initializes Input to Hidden below
     **/
    for( j = 1 ; j <= num_hidden ; j++ ) { 
        for( i = 0 ; i <= num_input ; i++ ) { 
            //bias_dih[j] = 0.0;
            dw_IH[i][j] = 0.0 ;
            w_IH[i + w_ihX * j] = init_further() ;
            //bias_wih[j] = init;
        }
    }

    /**
     * Initializes hidden to output below
     **/
    for( k = 1 ; k <= num_output ; k ++ ) {
        for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j][k] = 0.0 ;  
            //bias_dho[k] = 0.0;          
            w_HO[j + w_hoX * k] = init_further();
            //bias_who[k] = init;
        }
    }
     
    /**
     * TRAINING
     * This is where the good stuff happens
     * We take in the training data, shake it up, then start with:
     * (1) Forward propagation
     * (2) Backpropagation
     **/
    for( n = 0 ; n < epoch ; n++) { 
        
        /**
         * Randomization
         **/
        for( p = 1 ; p <= num_training ; p++ ) {
            r_pattern[p] = p ;
        }
        for( p = 1 ; p <= num_training ; p++) {
            np = p + rando() * ( num_training + 1 - p ) ;
            op = r_pattern[p] ; r_pattern[p] = r_pattern[np] ; r_pattern[np] = op ;
        }

        err = 0.0 ; /* Set error to zero for each epoch */
        for( np = 0 ; np < num_training ; np++ ) {
            p = r_pattern[np];

            /**
             * Computation of hidden unit activations
             * Follows basic sigmoidal activation
             **/
            for( j = 1 ; j <= num_hidden ; j++ ) {
                s_hidden[p][j] = w_IH[0+w_ihX*j] ;
                for( i = 1 ; i <= num_input ; i++ ) {s_hidden[p][j] += input[p][i] * w_IH[i+w_ihX*j] ;}
                hidden[p+hiddenX*j] = sigmoid(s_hidden[p][j]);
            }

            /**
             * Computation of all output activations
             * follows basic sigmoidal activation
             **/
            for( k = 1 ; k <= num_output ; k++ ) {
                s_out[p][k] =  w_HO[0+w_hoX*k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    s_out[p][k] += hidden[p+hiddenX*j] * w_HO[j+w_hoX*k] ;
                }
                output[p+outputX*k] = sigmoid(s_out[p][k]);
                err += 0.5 * (target[p][k] - output[p+outputX*k]) * (target[p][k] - output[p+outputX*k]) ; 
                d_out[k] = (target[p][k] - output[p+outputX*k]) * output[p+outputX*k] * (1.0 - output[p+outputX*k]) ;
            }

            /**
             * Back propogation
             * Takes output and backpropagates to the input to find the error
             * Updates weights and biases while doing so
             **/
            for( j = 1 ; j <= num_hidden ; j++ ) {
                s_do[j] = 0.0 ;
                for( k =1 ; k <= num_output ; k++ ) {
                    s_do[j] += w_HO[j+w_hoX*k] * d_out[k] ;
                }
                d_hidden[j] = s_do[j] * hidden[p+hiddenX*j] * (1.0 - hidden[p+hiddenX*j]) ;
            }

            /**
             * Update weights of input to hidden layer
             **/
            for( j = 1 ; j <= num_hidden ; j++ ) {
                dw_IH[0][j] = eta * d_hidden[j] + alpha * dw_IH[0][j] ;
                w_IH[0+w_ihX*j] += dw_IH[0][j] ;
                for( i = 1 ; i <= num_input ; i++ ) { 
                    dw_IH[i][j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i][j];
                    w_IH[i+w_ihX*j] += dw_IH[i][j] ;
                }
            }

            /**
             * Update weights for hidden to output hidden layer weights
             **/
            for( k = 1 ; k <= num_output ; k ++ ) {
                dw_HO[0][k] = eta * d_out[k] + alpha * dw_HO[0][k] ;

                w_HO[0+w_hoX*k] += dw_HO[0][k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    dw_HO[j][k] = eta * hidden[p+hiddenX*j] * d_out[k] + alpha * dw_HO[j][k] ;
                    w_HO[j+w_hoX*k] += dw_HO[j][k] ;
                }
            }

        }

        if( (n%100) == 0 ) printf("\nEpoch %d :   err = %f", n, err) ;
        if( err < 0.0003 ){
            printf("Caught early at err %f", err);
            break;
        } /* stop learning when 'near enough' */
    }
    
    printf("\n\nNeural Net - EP %d\n\nPat\t", n) ;   /* print network outputs */
    for( i = 1 ; i <= num_input ; i++ ) {
        printf("Input-4%d\t", i) ;
    }
    for( k = 1 ; k <= num_output ; k++ ) {
        printf("target%-4d\tnnet output%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= num_training ; p++ ) {        
    printf("\n%d\t", p) ;
        for( i = 1 ; i <= num_input ; i++ ) {
            printf("%f\t", input[p][i]) ;
        }
        for( k = 1 ; k <= num_output ; k++ ) {
            printf("%f\t%f\t", target[p][k], output[p+outputX*k]) ;
        }
    }
    printf("\n");
    copy_array(nnet->hidden, hidden, ((num_training+1)*(num_hidden+1)));
    copy_array(nnet->hidden_weights, w_IH, ((num_input+1)*(num_hidden+1)));
    copy_array(nnet->output_weights,  w_HO, ((num_hidden+1)*(num_output+1)));
    //nnet->lr = lr;
    copy_array(nnet->output, output, ((num_training+1)*(num_output+1)));
    print_all(nnet);
    return nnet;
}



/* ************** HELPER FUNCTIONS ***************** */
void free_nnet(cann_double *in){
    free(in->output_weights);
    free(in->hidden_weights);
    free(in->output);
    free(in->hidden);
    free(in->dw_HO);
    free(in->dw_IH);
    free(in->d_hidden);
    free(in->s_do);
    free(in->d_out);
    free(in->s_out);
    free(in->s_hidden);
    free(in);
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
    }

double d_sigmoid(double x){
    return sigmoid(x) * (1.0f - sigmoid(x));
}

double tanh_(double x){
    return 2 / 1 + exp(-2*x) - 1;
}

double d_tanh(double x){
    return 1 - pow((2/1+exp(-2*x) -1), 2);
}

double init_further(){
    return 2.0 * (init_weights() - 0.5) * 0.5;
}

double init_weights(){
    return ((double) rand())/((double) RAND_MAX);
}

void print_array(double input[], int length){
    printf("\n");
    for (int i=0; i < length; i++){
        printf("|%f|\n", input[i]);
    }
    printf("\n");
}

void print_mat(double input[], int lengthX, int lengthY){
    
    for (int x=0; x < lengthX; x++){
        printf("| ");
        for (int y=0; y < lengthY; y++){
            printf("%-11f ", input[x * lengthX + y]);
        }
        printf("|\n");
    }
    printf("\n");
}

double *init_zero(double input[], int length){
    for (int x=0; x < length; x++){
        input[x] = (double) 0;
    }
    return input;
}

double *init_random(double input[], int length){
    for (int x=0; x < length; x++){
        input[x] = init_weights();
    }
    return input;
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

void copy_array(double arr2[], double arr1[], int size){
    for(int i=0; i < size; i++){
        arr2[i] = arr1[i];
    }
}

void print_all(cann_double *nnet){
    printf("\n\n########## MODEL SUMMARY ##########\n");
    printf("NUM input: %d        NUM hidden: %d\nNUM output: %d\n", nnet->num_inputs, nnet->num_hidden, nnet->num_outputs);
    printf("----------VECTORS----------\nHidden:\n");
    print_mat(nnet->hidden, (nnet->num_training+1), (nnet->num_hidden+1));
    printf("output\n");
    print_mat(nnet->output, (nnet->num_training+1), (nnet->num_outputs));
    printf("----------WEIGHTS----------:\n");
    printf("hidden: \n");
    print_mat(nnet->hidden_weights, nnet->num_inputs+1, nnet->num_hidden+1);
    printf("output: \n");
    print_mat(nnet->output_weights, nnet->num_hidden+1, nnet->num_outputs+1);
    printf("----------BIASES-----------\n");
    printf("hidden BIAS: \n");
    //print_array(nnet->hiddenBias, nnet->num_hidden);
    printf("FINAL output BIAS: \n");
    print_array(nnet->output_weights, nnet->num_outputs);
}

