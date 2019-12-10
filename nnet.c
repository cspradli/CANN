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

cann_double *init_model_double(int num_inputs, int num_hidden, int num_outputs){


    cann_double* nnet = (cann_double* )malloc(sizeof(cann_double));
    if (!nnet) return 0;

    nnet->num_inputs = num_inputs;
    nnet->num_hidden = num_hidden;
    nnet->num_inputs = num_inputs;
    nnet->num_hidden = num_hidden;
    nnet->num_outputs = num_outputs;

    nnet->hidden = (double *) calloc(num_hidden, sizeof(double));
    nnet->output = (double *) calloc(num_outputs, sizeof(double));
    nnet->hiddenBias = (double *) calloc(num_hidden, sizeof(double));
    nnet->outBias = (double *) calloc(num_outputs,  sizeof(double));
    nnet->hidden_weights = (double *) calloc((num_inputs+num_hidden), sizeof(double));
    nnet->output_weights = (double *) calloc((num_hidden+num_outputs), sizeof(double));

    nnet->hidden = init_zero(nnet->hidden, num_hidden);
    nnet->output = init_zero(nnet->output, num_outputs);
    nnet->hiddenBias = init_random(nnet->hiddenBias, num_hidden);
    nnet->outBias = init_random(nnet->outBias, num_outputs);
    nnet->hidden_weights = init_random(nnet->hidden_weights, (num_inputs+num_hidden));
    nnet->output_weights = init_random(nnet->output_weights, (num_hidden+num_outputs));
    return nnet;

}

cann_double *model_fit(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training+1];
    double s_hidden[num_training+1][num_hidden+1], w_IH[num_input+1][num_hidden+1], hidden[num_training+1][num_hidden+1];
    double s_out[num_training+1][num_output+1], w_HO[num_hidden+1][num_output+1], output[num_training+1][num_output+1];
    double d_out[num_output+1], s_do[num_hidden+1], d_hidden[num_hidden+1];
    double dw_IH[num_input+1][num_hidden+1], dw_HO[num_hidden+1][num_output+1];
    double err, eta = 0.5, alpha = 0.9;
  

    // INITIALIZATION
    for( j = 1 ; j <= num_hidden ; j++ ) {    /* initialize w_IH and dw_IH */
        for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i][j] = 0.0 ;
            w_IH[i][j] = init_further() ;
        }
    }
    for( k = 1 ; k <= num_output ; k ++ ) {    /* initialize w_HO and dw_HO */
        for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j][k] = 0.0 ;              
            w_HO[j][k] = init_further() ;
        }
    }
     
    for( n = 0 ; n < epoch ; n++) { 
           /* iterate weight updates */
        for( p = 1 ; p <= num_training ; p++ ) {    /* randomize order of training patterns */
            r_pattern[p] = p ;
        }

        for( p = 1 ; p <= num_training ; p++) {
            np = p + rando() * ( num_training + 1 - p ) ;
            op = r_pattern[p] ; r_pattern[p] = r_pattern[np] ; r_pattern[np] = op ;
        }

        err = 0.0 ;
        for( np = 1 ; np <= num_training ; np++ ) {    /* repeat for all the training patterns */
            p = r_pattern[np];

            for( j = 1 ; j <= num_hidden ; j++ ) {    /* compute hidden unit activations */
                s_hidden[p][j] = w_IH[0][j] ;
                for( i = 1 ; i <= num_input ; i++ ) {
                    s_hidden[p][j] += input[p][i] * w_IH[i][j] ;
                }
                hidden[p][j] = 1.0/(1.0 + exp(-s_hidden[p][j])) ;
            }


            for( k = 1 ; k <= num_output ; k++ ) {
                s_out[p][k] = w_HO[0][k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    s_out[p][k] += hidden[p][j] * w_HO[j][k] ;
                }
                output[p][k] = 1.0/(1.0 + exp(-s_out[p][k])) ;
                err += 0.5 * (target[p][k] - output[p][k]) * (target[p][k] - output[p][k]) ; 

                d_out[k] = (target[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]) ;
            }


            for( j = 1 ; j <= num_hidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                s_do[j] = 0.0 ;
                for( k = 1 ; k <= num_output ; k++ ) {
                    s_do[j] += w_HO[j][k] * d_out[k] ;
                }
                d_hidden[j] = s_do[j] * hidden[p][j] * (1.0 - hidden[p][j]) ;
            }


            for( j = 1 ; j <= num_hidden ; j++ ) {     /* update weights w_IH */
                dw_IH[0][j] = eta * d_hidden[j] + alpha * dw_IH[0][j] ;
                w_IH[0][j] += dw_IH[0][j] ;
                for( i = 1 ; i <= num_input ; i++ ) { 
                    dw_IH[i][j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i][j];
                    w_IH[i][j] += dw_IH[i][j] ;
                }
            }


            for( k = 1 ; k <= num_output ; k ++ ) {    /* update weights w_HO */
                dw_HO[0][k] = eta * d_out[k] + alpha * dw_HO[0][k] ;
                w_HO[0][k] += dw_HO[0][k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    dw_HO[j][k] = eta * hidden[p][j] * d_out[k] + alpha * dw_HO[j][k] ;
                    w_HO[j][k] += dw_HO[j][k] ;
                }
            }

        }

        if( n%100 == 0 ) printf("\nEpoch %-5d :   err = %f", n, err) ;
        if( err < 0.0003 ){
            printf("Caught early at err %f", err);
            break;
        } /* stop learning when 'near enough' */
    }
    
    printf("\n\nNETWORK DATA - EPOCH %d\n\nPat\t", n) ;   /* print network outputs */
    for( i = 1 ; i <= num_input ; i++ ) {
        printf("Input%-4d\t", i) ;
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
            printf("%f\t%f\t", target[p][k], output[p][k]) ;
        }
    }
    printf("\n");
    return nnet;
}


cann_double *model_fit_update(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training+1];
    double s_hidden[num_training+1][num_hidden+1], w_IH[num_input+1][num_hidden+1], hidden[num_training+1][num_hidden+1];
    double s_out[num_training+1][num_output+1], w_HO[num_hidden+1][num_output+1], output[num_training+1][num_output+1];
    double d_out[num_output+1], s_do[num_hidden+1], d_hidden[num_hidden+1];
    double dw_IH[num_input+1][num_hidden+1], dw_HO[num_hidden+1][num_output+1];
    double err, eta = 0.5, alpha = 0.9;
  

    // INITIALIZATION
    for( j = 1 ; j <= num_hidden ; j++ ) {    /* initialize w_IH and dw_IH */
        for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i][j] = 0.0 ;
            w_IH[i][j] = init_further() ;
        }
    }
    for( k = 1 ; k <= num_output ; k ++ ) {    /* initialize w_HO and dw_HO */
        for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j][k] = 0.0 ;              
            w_HO[j][k] = init_further() ;
        }
    }
     
    for( n = 0 ; n < epoch ; n++) { 
           /* iterate weight updates */
        for( p = 1 ; p <= num_training ; p++ ) {    /* randomize order of training patterns */
            r_pattern[p] = p ;
        }

        for( p = 1 ; p <= num_training ; p++) {
            np = p + rando() * ( num_training + 1 - p ) ;
            op = r_pattern[p] ; r_pattern[p] = r_pattern[np] ; r_pattern[np] = op ;
        }

        err = 0.0 ;
        for( np = 1 ; np <= num_training ; np++ ) {    /* repeat for all the training patterns */
            p = r_pattern[np];

            for( j = 1 ; j <= num_hidden ; j++ ) {    /* compute hidden unit activations */
                s_hidden[p][j] = w_IH[0][j] ;
                for( i = 1 ; i <= num_input ; i++ ) {
                    s_hidden[p][j] += input[p][i] * w_IH[i][j] ;
                }
                hidden[p][j] = 1.0/(1.0 + exp(-s_hidden[p][j])) ;
            }


            for( k = 1 ; k <= num_output ; k++ ) {
                s_out[p][k] = w_HO[0][k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    s_out[p][k] += hidden[p][j] * w_HO[j][k] ;
                }
                output[p][k] = 1.0/(1.0 + exp(-s_out[p][k])) ;
                err += 0.5 * (target[p][k] - output[p][k]) * (target[p][k] - output[p][k]) ; 

                d_out[k] = (target[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]) ;
            }


            for( j = 1 ; j <= num_hidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                s_do[j] = 0.0 ;
                for( k = 1 ; k <= num_output ; k++ ) {
                    s_do[j] += w_HO[j][k] * d_out[k] ;
                }
                d_hidden[j] = s_do[j] * hidden[p][j] * (1.0 - hidden[p][j]) ;
            }


            for( j = 1 ; j <= num_hidden ; j++ ) {     /* update weights w_IH */
                dw_IH[0][j] = eta * d_hidden[j] + alpha * dw_IH[0][j] ;
                w_IH[0][j] += dw_IH[0][j] ;
                for( i = 1 ; i <= num_input ; i++ ) { 
                    dw_IH[i][j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i][j];
                    w_IH[i][j] += dw_IH[i][j] ;
                }
            }


            for( k = 1 ; k <= num_output ; k ++ ) {    /* update weights w_HO */
                dw_HO[0][k] = eta * d_out[k] + alpha * dw_HO[0][k] ;
                w_HO[0][k] += dw_HO[0][k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    dw_HO[j][k] = eta * hidden[p][j] * d_out[k] + alpha * dw_HO[j][k] ;
                    w_HO[j][k] += dw_HO[j][k] ;
                }
            }

        }

        if( n%100 == 0 ) printf("\nEpoch %-5d :   err = %f", n, err) ;
        if( err < 0.0003 ){
            printf("Caught early at err %f", err);
            break;
        } /* stop learning when 'near enough' */
    }
    
    printf("\n\nNETWORK DATA - EPOCH %d\n\nPat\t", n) ;   /* print network outputs */
    for( i = 1 ; i <= num_input ; i++ ) {
        printf("Input%-4d\t", i) ;
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
            printf("%f\t%f\t", target[p][k], output[p][k]) ;
        }
    }
    printf("\n");
    return nnet;
}




/* ************** HELPER FUNCTIONS ***************** */
void free_nnet(cann_double *in){
    free(in->output_weights);
    free(in->hidden_weights);
    free(in->outBias);
    free(in->hiddenBias);
    free(in->output);
    free(in->hidden);
    free(in);
}

double partial_dError(double x, double y){
    return x-y;
}



double err(double x, double y){
    return 0.5f * (x-y) * (x-y);
}

double toterror(double* tg, double* o, int size){
    double sum = 0.0f;
    for(int i=0; i<size;i++){
        sum += err(tg[i], o[i]);
    }
    return sum;
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
            printf("%f ", input[x * lengthX + y]);
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

void print_all(cann_double *nnet){
    printf("\n\n########## MODEL SUMMARY ##########\n");
    printf("NUM input: %d        NUM hidden: %d\nNUM output: %d\n", nnet->num_inputs, nnet->num_hidden, nnet->num_outputs);
    printf("----------VECTORS----------\nHidden:\n");
    print_array(nnet->hidden, 2);
    printf("output\n");
    print_array(nnet->output, nnet->num_outputs);
    printf("----------WEIGHTS----------:\n");
    printf("hidden: \n");
    print_mat(nnet->hidden_weights, nnet->num_inputs, nnet->num_hidden);
    printf("output: \n");
    print_mat(nnet->output_weights, nnet->num_hidden, nnet->num_outputs);
    printf("----------BIASES-----------\n");
    printf("hidden BIAS: \n");
    print_array(nnet->hiddenBias, nnet->num_hidden);
    printf("FINAL output BIAS: \n");
    print_array(nnet->output_weights, nnet->num_outputs);
}

