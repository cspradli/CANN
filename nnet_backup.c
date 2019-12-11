#include "nnet_backup.h"

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

cann_double *model_train(cann_double *nnet, int num_inputs, int num_hidden, int num_outputs, int num_training, double lr, int epochs, int training_order[], double training_in[], double training_out[]){
    double *hiddenLayer;
    double *outputLayer;
    double *hiddenLayerBias;
    double *outputLayerBias;
    double *hiddenWeights;
    double *outputWeights;
    hiddenLayer = nnet->hidden;
    outputLayer = nnet->output;
    hiddenLayerBias = nnet->hiddenBias;
    outputLayerBias = nnet->outBias;
    hiddenWeights = nnet->hidden_weights;
    outputWeights = nnet->output_weights;

    for (int n = 0; n < epochs; n++){

        /**
         * Need to shuffle incoming trainign order for max stochasticity
         **/
        shuffle(training_order, num_training);
        
        for (int x = 0; x < num_training; x++){
            int i = training_order[x];
            
            
            //Compute hidden layer activation
            for (int j = 0; j < num_hidden; j++){
                double activation = hiddenLayerBias[j];         // nnet->hiddenBias[j];
                for (int k = 0; k < num_inputs; k++){
                    activation = activation + 
                    training_in[i*num_training +k]*
                    nnet->hidden_weights[k*num_hidden+j];       //nnet->hidden_weights[k*num_hidden+j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            //print_array(hiddenLayer, num_hidden);
            
            
            //Comput output layer activation
            for (int j=0; j<num_outputs;j++){                   //J = num of output nodes
                double activation = outputLayerBias[j];
                for(int k=0; k<num_hidden;k++){                 // k = num of hidden nodes
                    activation += hiddenLayer[k]*
                    outputWeights[k*num_outputs + j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            
            /**
             * BAKCPROP
             * Next steps involve:
             * Calculating incremental changre in network weights
             * Moves network towards minimizing the err of the output
             * Starts at the node and works itself backward
             **/
            double deltaOut[num_outputs];
            for (int j=0; j<num_outputs; j++){
                double derivative_error = (training_out[i*num_training+j]-outputLayer[j]);
                deltaOut[j] = derivative_error * d_sigmoid(outputLayer[j]);
                if ((n%1000)==0) printf("EPOCH: %d\nDERIV OF MSE: %f\n", n,derivative_error);
            }

            /**
             * hidden layer backprop
             * err calculation for given node = sum of err across all output nodes
             **/
            double delta_hidden[num_hidden];
            for (int j=0; j<num_hidden; j++){
                double deriv_error = 0.0f;
                for (int k=0; k<num_outputs; k++){
                    deriv_error += deltaOut[k] * outputWeights[j*num_outputs+k];
                }
                delta_hidden[j] = deriv_error*d_sigmoid(hiddenLayer[j]);
            }

            /**
             * Apply deltas to respective weight matrices
             * in addition to bias units
             * 1st: Apply change in output weights
             * 2nd: Apply change in hidden weights
             **/
            for (int j=0; j<num_outputs;j++){
                outputLayerBias[j] += deltaOut[j]*lr;
                for(int k=0; k<num_hidden; k++){
                    outputWeights[k*num_outputs+j] += hiddenLayer[k]*deltaOut[j]*lr;
                }
            }
            for (int j=0; j<num_hidden; j++){
                hiddenLayerBias[j] += delta_hidden[j]*lr;
                for (int k=0; k<num_inputs; k++){
                    hiddenWeights[k*num_hidden+j] += training_in[i*num_training+k]*delta_hidden[j]*lr;
                }
            }


        }
    }
    print_array(hiddenLayer, num_hidden);
    print_mat(hiddenWeights, num_inputs, num_hidden);
    nnet->hidden = hiddenLayer;
    nnet->hidden_weights = hiddenWeights;
    nnet->hiddenBias = hiddenLayerBias;
    nnet->output = outputLayer;
    nnet->output_weights = outputWeights;
    nnet->outBias = outputLayerBias;
    return nnet;

}

cann_double *train(cann_double *nnet, int num_inputs, int num_hidden, int num_outputs, int num_training, double lr, int epochs, int trainingOrder[], double training_in[], double training_out[]){
    for (int i=0; i<epochs; i++){
        shuffle(trainingOrder, num_training);
        double err = 0.0f;
        for(int j =0; j<num_training;j++){
            forward_prop(nnet, training_in);
            backprop(nnet, training_in, training_out, lr);
            err += toterror(training_out, nnet->output, nnet->num_outputs);
        }
        printf("err %.12f || learning rate %f\n", (err/num_training), lr);
        //lr *= 0.99;
    }
    return nnet;
}

double *predict(cann_double *nnet, double *in){
    forward_prop(nnet, in);
    return nnet->output;
}

void forward_prop(cann_double *nnet, double *in){
    for (int j = 0; j < nnet->num_hidden; j++){
        
        double activation = nnet->hiddenBias[j];
        
        for (int k = 0; k < nnet->num_inputs; k++){
            activation += in[k]*nnet->hidden_weights[j*nnet->num_inputs+k];       //nnet->hidden_weights[k*num_hidden+j];
        }
        
        nnet->hidden[j] = sigmoid(activation);
    }

    for (int j=0; j<nnet->num_outputs;j++){                   //J = num of output nodes
        double activation = nnet->outBias[j];

        for(int k=0; k<nnet->num_hidden;k++){                 // k = num of hidden nodes
            activation += nnet->hidden[k]*nnet->output_weights[j*nnet->num_outputs+k];
        }

        nnet->output[j] = sigmoid(activation);
    }

    //print_array(nnet->output, nnet->num_outputs);

}

void backprop(cann_double *nnet, double *training_input, double *training_out, double rate){
            /**
             * BAKCPROP
             * Next steps involve:
             * Calculating incremental changre in network weights
             * Moves network towards minimizing the err of the output
             * Starts at the node and works itself backward
             **/
    for (int i=0; i < nnet->num_hidden; i++){
        double activation = 0.0f;
        for (int j=0; j<nnet->num_outputs;j++){
            double x = partial_dError(nnet->output[j], training_out[j]);
            double y = d_sigmoid(nnet->output[j]);
            activation += x * y * nnet->output_weights[j*nnet->num_hidden+i];
            nnet->output_weights[j * nnet->num_inputs + j] -= rate  * activation * d_sigmoid(nnet->hidden[i])*training_input[j];
        }
    }
}


cann_double *model_fit_update(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input], double target[][num_output], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training];
    
    double s_hidden[num_training][num_hidden];
    double s_out[num_training][num_output];
    double d_out[num_output], s_do[num_hidden], d_hidden[num_hidden];
    double dw_IH[num_input][num_hidden], dw_HO[num_hidden][num_output];

    double hidden[((num_training)*(num_hidden))];
    double output[((num_training)*(num_output))];
    double w_IH[((num_input)*(num_hidden))];
    double w_HO[((num_hidden)*(num_output))];
    
    double bias_dih[num_hidden], bias_who[num_output], bias_dho[num_output], bias_wih[num_hidden];

    double err, eta = 0.5, alpha = 0.9;
    
    int hiddenX;
    int outputX;
    int w_ihX;
    int w_hoX;
    
    hiddenX = (num_training);
    outputX = (num_training);
    w_ihX = (num_input);
    w_hoX = (num_hidden);

    // INITIALIZATION
    for( j = 0 ; j < num_hidden ; j++ ) {    /* initialize w_IH and dw_IH */
        for( i = 0 ; i < num_input ; i++ ) { 
            double init = init_further();
            bias_dih[j] = 0.0;
            dw_IH[i][j] = 0.0 ;
            w_IH[i + w_ihX * j] = init ;
            bias_wih[j] = init;
        }
    }
    for( k = 0 ; k < num_output ; k ++ ) {    /* initialize w_HO and dw_HO */
        for( j = 0 ; j < num_hidden ; j++ ) {
            double init = init_further();
            
            dw_HO[j][k] = 0.0 ;  
            bias_dho[k] = 0.0;          
            
            w_HO[j + w_hoX * k] = init;
            bias_who[k] = init;
        }
    }
     
    for( n = 0 ; n < epoch ; n++) { 
           /* iterate weight updates */
        for( p = 0 ; p < num_training ; p++ ) {    /* randomize order of training patterns */
            r_pattern[p] = p ;
        }

        for( p = 0 ; p < num_training ; p++) {
            np = p + rando() * ( num_training - p ) ;
            op = r_pattern[p] ; r_pattern[p] = r_pattern[np] ; r_pattern[np] = op ;
        }

        err = 0.0 ;
        for( np = 0 ; np < num_training ; np++ ) {    /* repeat for all the training patterns */
            p = r_pattern[np];

            for( j = 0 ; j < num_hidden ; j++ ) {    /* compute hidden unit activations */
                //s_hidden[p][j] = w_IH[0+w_ihX*j] ;
                s_hidden[p][j] = bias_wih[j];
                for( i = 0 ; i < num_input ; i++ ) {
                    s_hidden[p][j] += input[p][i] * w_IH[i+w_ihX*j] ;
                }
                hidden[p+hiddenX*j] = 1.0/(1.0 + exp(-s_hidden[p][j])) ;
            }


            for( k = 0 ; k < num_output ; k++ ) {
                s_out[p][k] = bias_who[k];// w_HO[0+w_hoX*k] ;
                for( j = 0 ; j < num_hidden ; j++ ) {
                    s_out[p][k] += hidden[p+hiddenX*j] * w_HO[j+w_hoX*k] ;
                }
                output[p+outputX*k] = 1.0/(1.0 + exp(-s_out[p][k])) ;
                err += 0.5 * (target[p][k] - output[p+outputX*k]) * (target[p][k] - output[p+outputX*k]) ; 

                d_out[k] = (target[p][k] - output[p+outputX*k]) * output[p+outputX*k] * (1.0 - output[p+outputX*k]) ;
            }


            for( j = 0 ; j < num_hidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                s_do[j] = 0.0 ;
                for( k =0 ; k < num_output ; k++ ) {
                    s_do[j] += w_HO[j+w_hoX*k] * d_out[k] ;
                }
                d_hidden[j] = s_do[j] * hidden[p+hiddenX*j] * (1.0 - hidden[p+hiddenX*j]) ;
            }


            for( j = 0 ; j < num_hidden ; j++ ) {     /* update weights w_IH */
                //dw_IH[0][j] = eta * d_hidden[j] + alpha * dw_IH[0][j] ;
                bias_dih[j] = eta * d_hidden[j] + alpha * bias_dih[j];
                //w_IH[0+w_ihX*j] += activation ;
                bias_wih[j] += bias_dih[j];
                for( i = 0 ; i < num_input ; i++ ) { 
                    dw_IH[i][j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i][j];
                    w_IH[i+w_ihX*j] += dw_IH[i][j] ;
                }
            }


            for( k = 0 ; k < num_output ; k ++ ) {    /* update weights w_HO */
                //dw_HO[0][k] = eta * d_out[k] + alpha * dw_HO[0][k] ;

                bias_dho[k] = eta * d_out[k] + alpha * bias_dho[k];
                //w_HO[0+w_hoX*k] += dw_HO[0][k] ;
                bias_who[k] += bias_dho[k];
                for( j = 0 ; j < num_hidden ; j++ ) {
                    dw_HO[j][k] = eta * hidden[p+hiddenX*j] * d_out[k] + alpha * dw_HO[j][k] ;
                    w_HO[j+w_hoX*k] += dw_HO[j][k] ;
                }
            }

        }

        if( n%100 == 0 ) printf("\nEpoch %-5d :   err = %f", n, err) ;
        if( err < 0.0003 ){
            printf("Caught early at err %f", err);
            break;
        } /* stop learning when 'near enough' */
    }
    
    printf("\n\nNeural Net - EP %d\n\nPat\t", n) ;   /* print network outputs */
    for( i = 0 ; i < num_input ; i++ ) {
        printf("Input-4%d\t", i) ;
    }
    for( k = 0 ; k < num_output ; k++ ) {
        printf("target%-4d\tnnet output%-4d\t", k, k) ;
    }
    for( p = 0 ; p < num_training ; p++ ) {        
    printf("\n%d\t", p) ;
        for( i = 0 ; i < num_input ; i++ ) {
            printf("%f\t", input[p][i]) ;
        }
        for( k = 0 ; k < num_output ; k++ ) {
            printf("%f\t%f\t", target[p][k], output[p+outputX*k]) ;
        }
    }
    printf("\n");
    copy_array(nnet->hidden, hidden, ((num_training)*(num_hidden)));
    copy_array(nnet->hidden_weights, w_IH, ((num_input)*(num_hidden)));
    copy_array(nnet->output_weights,  w_HO, ((num_hidden)*(num_output)));
    //nnet->lr = lr;
    copy_array(nnet->output, output, ((num_training)*(num_output)));
    
    return nnet;
}

// MOST RECENT 

/**
 * NNET portion of ANN - C
 * Author Caleb Spradlin
 **/

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "nnet.h"
#include "nnet_io.h"

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;


#define rando() ((double)rand()/((double)RAND_MAX+1))

cann_double *init_model_double(int num_training, int num_inputs, int num_hidden, int num_outputs, int prev_trained){


    cann_double* nnet = (cann_double* )malloc(sizeof(cann_double));
    if (!nnet) return 0;

    nnet->num_inputs   = num_inputs;
    nnet->num_hidden   = num_hidden;
    nnet->num_inputs   = num_inputs;
    nnet->num_hidden   = num_hidden;
    nnet->num_outputs  = num_outputs;
    nnet->num_training = num_training;
    nnet->prev_trained = prev_trained;
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

    double err, eta = 0.01, alpha = 0.9;
    double error_out = 0.0;
    int hiddenX, outputX, w_ihX, w_hoX;

    hiddenX = (num_training+1); 
    outputX = (num_training+1); 
    w_ihX = (num_input+1);
    w_hoX = (num_hidden+1);

    if (nnet->prev_trained == 1){
        printf("Working with pre-trained model\n");
        copy_array(hidden, nnet->hidden, ((num_training+1)*(num_hidden+1)));
        copy_array(output, nnet->output, ((num_training+1)*(num_output+1)));
        copy_array(w_IH, nnet->hidden_weights, ((num_input+1)*(num_hidden+1)));
        copy_array(w_HO, nnet->output_weights, ((num_hidden+1)*(num_output+1)));
        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i][j] = 0.0 ;
            }
        }
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j][k] = 0.0 ;  
            }
        }

    } else {
        /**
        * Initialization of all weights and derivative of weights
        * Initializes Input to Hidden below
        **/
        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i][j] = 0.0 ;
            w_IH[i + w_ihX * j] = init_further() ;
            }
        }

        /**
        * Initializes hidden to output below
        **/
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
                dw_HO[j][k] = 0.0 ;          
                w_HO[j + w_hoX * k] = init_further();
            }
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
            printf("Caught early at err %f\n", err);
            break;
        } 


    }
    
    printf("\n\nNeural Net - EP %d\n\nPat\t", n) ;
    for( i = 1 ; i <= num_input ; i++ ) { printf("Input%-4d\t", i) ;}
    for( k = 1 ; k <= num_output ; k++ ) { printf("target%-4d\tnnet output%-4d\t", k, k) ; }
    
    for( p = 1 ; p <= num_training ; p++ ) {        
    
        printf("\n%d\t", p);
        for( i = 1 ; i <= num_input ; i++ ) {
            printf("%f\t", input[p][i]) ;
        }
        for( k = 1 ; k <= num_output ; k++ ) {
            printf("%f\t%f\t", target[p][k], output[p+outputX*k]) ;
        }
    
    }
    
    pthread_t tid[4];
    struct arg targ[4];
    for (i=0; i<4; i++){
        if (i==0){
            targ[i].a = nnet->hidden;
            targ[i].b = hidden;
            targ[i].size= ((num_training+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i ==1){
            targ[i].a = nnet->hidden_weights;
            targ[i].b = w_IH;
            targ[i].size= ((num_input+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==2){
            targ[i].a = nnet->output_weights;
            targ[i].b = w_HO;
            targ[i].size= ((num_hidden+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==3){
            targ[i].a = nnet->output;
            targ[i].b = output;
            targ[i].size= ((num_training+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        }
    }

    //nnet->hidden = hidden;
    //nnet->hidden_weights = w_IH;
    //nnet->output_weights = w_HO;
    //nnet->output = output;
    //copy_array(nnet->hidden, hidden, ((num_training+1)*(num_hidden+1)));
    //copy_array(nnet->hidden_weights, w_IH, ((num_input+1)*(num_hidden+1)));
    //copy_array(nnet->output_weights,  w_HO, ((num_hidden+1)*(num_output+1)));
    //copy_array(nnet->output, output, ((num_training+1)*(num_output+1)));
    nnet->err = error_out;
    return nnet;
}


void *thread(void *targ){
    struct arg* arg = (struct arg*) targ;
    //pthread_detach(pthread_self());
    pthread_mutex_lock(&mutex1);
    //rintf("in here as %lu\n", pthread_self());
    copy_array( arg->a, arg->b, arg->size);
    pthread_mutex_lock(&mutex1);
    pthread_exit(0);
    //return NULL;
}
/*
void *spin_thread(void *larg){
    printf("yee haw\n");
    struct arg* arg = (struct arg*) larg;
    printf("<<<<<<<<<<<<<<<<<<< Thread created connection = %s >>>>>>>>>>>>>>>>>>\n", arg->argv[2]);
    pthread_detach(pthread_self());
    printf("Running with server\n");
    get_input(arg->argc, arg->argv);
    pthread_exit(0);
}*/

/*void multi_train(int num_threads,int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    pthread_t tid[num_threads];
    struct arg targ[num_threads];
    cann_double *nets[num_threads];
    int epoch_in = epoch;
    double err = 100;
    double *erray[num_threads];
    double lr_in = 0.1;
    int num_r = 1;
    for(int i=0; i<num_threads;i++){
        nets[i] = init_model_double(num_training, num_input, num_hidden, num_output, 0);
        targ[i].epoch = epoch_in;
        targ[i].err = err;
        targ[i].lr = lr_in;
        targ[i].my_data = nets[i];
        targ[i].num_rows = num_r;
        pthread_create(&tid[i], NULL, thread, (void *)&targ[i]);
        printf("pthread_created %d\n", i);
    }
    //sleep(2);
    for (int i=0; i<num_threads; i++){
        pthread_join(tid[i], (void **)&(erray[i]));
    }

}*/


/* ************** HELPER FUNCTIONS ***************** */

void free_nnet(cann_double *in){
    free(in->output_weights);
    free(in->hidden_weights);
    free(in->output);
    free(in->hidden);
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


cann_double *model_fit(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training+1];
    double s_hidden[num_training+1][num_hidden+1];
    double s_out[num_training+1][num_output+1];
    double d_out[num_output+1], s_do[num_hidden+1], d_hidden[num_hidden+1];
    double dw_IH[(num_input+1)*(num_hidden+1)], dw_HO[(num_hidden+1)*(num_output+1)];
    double hidden[((num_training+1)*(num_hidden+1))], output[((num_training+1)*(num_output+1))];
    double w_IH[((num_input+1)*(num_hidden+1))], w_HO[((num_hidden+1)*(num_output+1))];
    
    //double bias_dih[num_hidden+1], bias_who[num_output+1], bias_dho[num_output+1], bias_wih[num_hidden+1];

    double err, eta = 0.01, alpha = 0.9;
    double error_out = 0.0;
    int hiddenX, outputX, w_ihX, w_hoX;

    hiddenX = (num_training+1); 
    outputX = (num_training+1); 
    w_ihX = (num_input+1);
    w_hoX = (num_hidden+1);

    if (nnet->prev_trained == 1){
        printf("Working with pre-trained model\n");
        copy_array(hidden, nnet->hidden, ((num_training+1)*(num_hidden+1)));
        copy_array(output, nnet->output, ((num_training+1)*(num_output+1)));
        copy_array(w_IH, nnet->hidden_weights, ((num_input+1)*(num_hidden+1)));
        copy_array(w_HO, nnet->output_weights, ((num_hidden+1)*(num_output+1)));
        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i+w_ihX*j] = 0.0 ;
            }
        }
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j+w_hoX*k] = 0.0 ;  
            }
        }

    } else {
        printf("Working with new model\n");
        /**
        * Initialization of all weights and derivative of weights
        * Initializes Input to Hidden below
        **/
        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i + w_ihX * j] = 0.0 ;
            w_IH[i + w_ihX * j] = init_further() ;
            }
        }

        /**
        * Initializes hidden to output below
        **/
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
                dw_HO[j + w_hoX + k] = 0.0 ;          
                w_HO[j + w_hoX * k] = init_further();
            }
        }
    }
    printf("Working with new model 2\n");

    /**
     * TRAINING
     * This is where the good stuff happens
     * We take in the training data, shake it up, then start with:
     * (1) Forward propagation
     * (2) Backpropagation
     **/
    for( n = 0 ; n < epoch ; n++) { 
        
        printf("Working with new model %d\n", n);
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
            printf("Working with new model %d|%d\n", n, np);

            /**
             * Computation of hidden unit activations
             * Follows basic sigmoidal activation
             **/
            for( j = 1 ; j <= num_hidden ; j++ ) {
                printf("Working with new model %d|%d\n", j, n);
                s_hidden[p][j] = w_IH[0 + w_ihX * j] ;
                printf("Working with new model 11%d|%d\n", n, np);
                for( i = 1 ; i <= num_input ; i++ ) {
                    s_hidden[p][j] += input[p][i] * w_IH[i + w_ihX * j] ;
                }
                hidden[p + hiddenX * j] = sigmoid(s_hidden[p][j]);
            }
            printf("Working with new model 2%d|%d\n", n, np);

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


            printf("Working with new model 3%d|%d\n", n, np);
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
                dw_IH[0+w_ihX*j] = eta * d_hidden[j] + alpha * dw_IH[0+w_ihX*j] ;
                w_IH[0+w_ihX*j] += dw_IH[0+w_ihX*j] ;
                for( i = 1 ; i <= num_input ; i++ ) { 
                    dw_IH[i+w_ihX*j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i + w_ihX * j];
                    w_IH[i+w_ihX*j] += dw_IH[i + w_ihX * j] ;
                }
            }



            /**
             * Update weights for hidden to output hidden layer weights
             **/
            for( k = 1 ; k <= num_output ; k ++ ) {
                dw_HO[0+w_hoX*k] = eta * d_out[k] + alpha * dw_HO[0+w_hoX*k] ;

                w_HO[0+w_hoX*k] += dw_HO[0+w_hoX*k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    dw_HO[j+w_hoX*k] = eta * hidden[p+hiddenX*j] * d_out[k] + alpha * dw_HO[j+w_hoX*k] ;
                    w_HO[j+w_hoX*k] += dw_HO[j + w_hoX * k] ;
                }
            }

        }



        if( (n%100) == 0 ) printf("\nEpoch %d :   err = %f", n, err) ;
        if( err < 0.0003 ){
            printf("Caught early at err %f\n", err);
            break;
        } 


    }
    
    printf("\n\nNeural Net - EP %d\n\nPat\t", n) ;
    for( i = 1 ; i <= num_input ; i++ ) { printf("Input%-4d\t", i) ;}
    for( k = 1 ; k <= num_output ; k++ ) { printf("target%-4d\tnnet output%-4d\t", k, k) ; }
    
    for( p = 1 ; p <= num_training ; p++ ) {        
    
        printf("\n%d\t", p);
        for( i = 1 ; i <= num_input ; i++ ) {
            printf("%f\t", input[p][i]) ;
        }
        for( k = 1 ; k <= num_output ; k++ ) {
            printf("%f\t%f\t", target[p][k], output[p+outputX*k]) ;
        }
    
    }
    
    /*pthread_t tid[4];
    struct arg targ[4];
    for (i=0; i<4; i++){
        if (i==0){
            targ[i].a = nnet->hidden;
            targ[i].b = hidden;
            targ[i].size= ((num_training+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i ==1){
            targ[i].a = nnet->hidden_weights;
            targ[i].b = w_IH;
            targ[i].size= ((num_input+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==2){
            targ[i].a = nnet->output_weights;
            targ[i].b = w_HO;
            targ[i].size= ((num_hidden+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==3){
            targ[i].a = nnet->output;
            targ[i].b = output;
            targ[i].size= ((num_training+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        }
    }*/

    //nnet->hidden = hidden;
    //nnet->hidden_weights = w_IH;
    //nnet->output_weights = w_HO;
    //nnet->output = output;
    copy_array(nnet->hidden, hidden, ((num_training+1)*(num_hidden+1)));
    copy_array(nnet->hidden_weights, w_IH, ((num_input+1)*(num_hidden+1)));
    copy_array(nnet->output_weights,  w_HO, ((num_hidden+1)*(num_output+1)));
    copy_array(nnet->output, output, ((num_training+1)*(num_output+1)));
    return nnet;
}

cann_double *model_fit(cann_double *nnet, int num_training, int num_input, int num_hidden, int num_output, double input[][num_input+1], double target[][num_output+1], int epoch, double lr){
    int    i ,n, j, k, p, np, op, r_pattern[num_training+1];
    double s_hidden[num_training+1][num_hidden+1];
    double s_out[num_training+1][num_output+1];
    double d_out[num_output+1], s_do[num_hidden+1], d_hidden[num_hidden+1];
    double dw_IH[((num_input+1)*(num_hidden+1))];
    double dw_HO[((num_hidden+1)*(num_output+1))];
    double hidden[((num_training+1)*(num_hidden+1))], output[((num_training+1)*(num_output+1))];
    double w_IH[((num_input+1)*(num_hidden+1))], w_HO[((num_hidden+1)*(num_output+1))];
    
    //double bias_dih[num_hidden+1], bias_who[num_output+1], bias_dho[num_output+1], bias_wih[num_hidden+1];

    double err, eta = 0.01, alpha = 0.9;
    
    int hiddenX, outputX, w_ihX, w_hoX;

    hiddenX = (num_training+1); 
    outputX = (num_training+1); 
    w_ihX = (num_input+1);
    w_hoX = (num_hidden+1);

    if (nnet->prev_trained == 1){

        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i+w_ihX*j] = 0.0 ;
            }
        }
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
            dw_HO[j+w_hoX*k] = 0.0 ;  
            }
        }

    } else {
        /**
        * Initialization of all weights and derivative of weights
        * Initializes Input to Hidden below
        **/
        for( j = 1 ; j <= num_hidden ; j++ ) { 
            for( i = 0 ; i <= num_input ; i++ ) { 
            dw_IH[i+w_ihX*j] = 0.0 ;
            w_IH[i + w_ihX * j] = init_further() ;
            }
        }

        /**
        * Initializes hidden to output below
        **/
        for( k = 1 ; k <= num_output ; k ++ ) {
            for( j = 0 ; j <= num_hidden ; j++ ) {
                dw_HO[j+w_hoX*k] = 0.0 ;          
                w_HO[j + w_hoX * k] = init_further();
            }
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
                dw_IH[0+w_ihX*j] = eta * d_hidden[j] + alpha * dw_IH[0+w_ihX*j] ;
                w_IH[0+w_ihX*j] += dw_IH[0+w_ihX*j] ;
                for( i = 1 ; i <= num_input ; i++ ) { 
                    dw_IH[i+w_ihX*j] = eta * input[p][i] * d_hidden[j] + alpha * dw_IH[i+w_ihX*j];
                    w_IH[i+w_ihX*j] += dw_IH[i+w_ihX*j] ;
                }
            }


            //print_array(dw_HO, (num_hidden+1)*(num_output+1));
            /**
             * Update weights for hidden to output hidden layer weights
             **/
            for( k = 1 ; k <= num_output ; k ++ ) {
                dw_IH[0+w_hoX*k] = eta * d_out[k] + alpha * dw_IH[0+w_hoX*k] ;

                w_HO[0+w_hoX*k] += dw_IH[0+w_hoX*k] ;
                for( j = 1 ; j <= num_hidden ; j++ ) {
                    dw_IH[j+w_hoX*k] = eta * hidden[p+hiddenX*j] * d_out[k] + alpha * dw_IH[j+w_hoX*k] ;
                    w_HO[j+w_hoX*k] += dw_IH[j+w_hoX*k] ;
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
    for( i = 1 ; i <= num_input ; i++ ) { printf("Input%-4d\t", i) ;}
    for( k = 1 ; k <= num_output ; k++ ) { printf("target%-4d\tnnet output%-4d\t", k, k) ; }
    
    for( p = 1 ; p <= num_training ; p++ ) {        
    
        printf("\n%d\t", p);
        for( i = 1 ; i <= num_input ; i++ ) {
            printf("%f\t", input[p][i]) ;
        }
        for( k = 1 ; k <= num_output ; k++ ) {
            printf("%f\t%f\t", target[p][k], output[p+outputX*k]) ;
        }
    
    }
    
    /*pthread_t tid[4];
    struct arg targ[4];
    for (i=0; i<4; i++){
        if (i==0){
            targ[i].a = nnet->hidden;
            targ[i].b = hidden;
            targ[i].size= ((num_training+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i ==1){
            targ[i].a = nnet->hidden_weights;
            targ[i].b = w_IH;
            targ[i].size= ((num_input+1)*(num_hidden+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==2){
            targ[i].a = nnet->output_weights;
            targ[i].b = w_HO;
            targ[i].size= ((num_hidden+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        } else if(i==3){
            targ[i].a = nnet->output;
            targ[i].b = output;
            targ[i].size= ((num_training+1)*(num_output+1));
            pthread_create(&tid[i], NULL, thread, (void *) &targ[i]);
        }
    }*/

    copy_array(nnet->hidden, hidden, ((num_training+1)*(num_hidden+1)));
    copy_array(nnet->hidden_weights, w_IH, ((num_input+1)*(num_hidden+1)));
    copy_array(nnet->output_weights,  w_HO, ((num_hidden+1)*(num_output+1)));
    copy_array(nnet->output, output, ((num_training+1)*(num_output+1)));
    return nnet;
}
