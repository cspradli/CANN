#include "nnet_io.h"

double **init_2d(int rows, int columns){
    double **r = (double **) malloc((rows) * sizeof(double*));
    for(int i=0; i < columns; i++){
        r[i] = (double*) malloc((columns) * sizeof(double));
    }
    return r;
}

void nnet_save(cann_double *nnet, char *path){
    FILE* const file = fopen(path, "w");
    printf("    Printing to file %s", path);
    int i;
    fprintf(file, "%d %d %d %d\n", nnet->num_inputs, nnet->num_hidden, nnet->num_outputs, nnet->num_training);
    for (i = 0; i < ((nnet->num_inputs+1)*(nnet->num_hidden+1)); i++) fprintf(file, "%f\n", (double) nnet->hidden_weights[i]);
    for (i = 0; i < ((nnet->num_hidden+1)*(nnet->num_outputs+1));i++) fprintf(file, "%f\n", (double) nnet->output_weights[i]);
    for (i = 0; i < ((nnet->num_training+1)*(nnet->num_hidden+1)); i++) fprintf(file, "%f\n", (double) nnet->hidden[i]);
    for (i = 0; i < ((nnet->num_hidden+1)*(nnet->num_outputs+1)); i++) fprintf(file, "%f\n", (double) nnet->output[i]);
    fclose(file);
}

cann_double *nnet_load(char *path){
    int i;
    FILE* const file = fopen(path, "r");
    printf("    Reading from path %s\n", path);
    int num_input = 0;
    int num_hidden = 0;
    int num_output = 0;
    int num_training = 0;
    if(!fscanf(file, "%d %d %d %d\n", &num_input, &num_hidden, &num_output, &num_training)) printf("yeehow\n");
    cann_double *nnet;
    nnet = init_model_double(num_training, num_input, num_hidden, num_output, 1);
    for (i = 0; i < ((num_input+1)*(num_hidden+1)); i++) fscanf(file, "%lf\n", &nnet->hidden_weights[i]);
    for (i = 0; i < ((num_hidden+1)*(num_output+1));i++) fscanf(file, "%lf\n", &nnet->output_weights[i]);
    for (i = 0; i < ((num_training+1)*(num_hidden+1)); i++) fscanf(file, "%lf\n", &nnet->hidden[i]);
    for (i = 0; i < ((num_hidden+1)*(num_output+1)); i++) fscanf(file, "%lf\n", &nnet->output[i]);
    fclose(file);
    return nnet;
}

void free_data(data *in){
    for (int i =0; i < in->num_rows; i++){
        free(in->target_in[i]);
        free(in->target[i]);
    }
    free(in->target_in);
    free(in->target);
    free(in);
}
