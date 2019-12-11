#include "nnet_io.h"

void check_paths(char *path){
    //pid_t pid;
    if(fork() == 0){
        char cmd[256];
        sprintf(cmd, "./my_touch ");
        strcat(cmd, path);
        if(system(cmd)){
            printf("ERROR\n");
        }
    }
    wait(NULL);
}

int get_lines(char *path){
    FILE *file = fopen(path, "r");
    int line_count = 1;
    char chr;
    if (file == NULL){
        printf("could not read from %s", path);
        return 0;
    }
    for (chr = getc(file); chr != EOF; chr = getc(file)){
        if (chr == '\n') line_count++;
    }
    fclose(file);
    return line_count;
}

char *get_ln(FILE* file){
    char ch;
    int r = 0;
    int buf_size = 128;
    char *line = (char *) malloc(buf_size * sizeof(char));
    while ((ch = getc(file)) != EOF && ch != '\n')
    {
        line[r++] = ch;
        if(r + 1 == buf_size){
            line = (char *) realloc((line), (buf_size*=2)*sizeof(char));
        }
        /* code */
    }
    line[r] = '\0';
    return line;
    
}

void parse_data(data *in, char* line, int row)
{
    int cols = ((in->num_input+1) + (in->num_output+1));
    for(int col = 0; col < cols; col++)
    {
        const double val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < (in->num_input+1)){
        //    printf("LESS THAN %d %d\n", col, row);
            in->target_in[row][col] = val;
         } else {
          //  printf("GREATER THAN %d %d\n", col, row);
            in->target[row][col - (in->num_input+1)] = val;
            //printf("GT2 %d %d\n", col, in->num_input);
         }
    }
}

data *get_data(char *path, int num_inputs, int num_outputs){
    FILE *file = fopen(path, "r");
    data *new_dat;
    if (!file){
        printf("No file at: %s\n", path);
    }
    
    int num_rows = get_lines(path);
    new_dat = (data *) malloc(sizeof(data));
    new_dat->target_in = init_2d(num_rows+5, num_inputs+5);
    new_dat->target = init_2d(num_rows+5, num_outputs+5);
    new_dat->num_input = num_inputs;
    new_dat->num_output = num_outputs;
    new_dat->num_rows = num_rows;
    
    for (int i=0; i < num_rows; i++){
        char *line = get_ln(file);
        printf("%s\n", line);
        parse_data(new_dat, line, num_rows);
        free(line);
    }

    fclose(file);
    return new_dat;
}

double **init_2d(int rows, int columns){
    double **r = (double **) malloc((rows) * sizeof(double*));
    for(int i=0; i < rows; i++){
        r[i] = (double*) malloc((columns) * sizeof(double));
    }
    return r;
}

void nnet_save(cann_double *nnet, char *path){
    check_paths(path);
    FILE* const file = fopen(path, "w");
    printf("    Printing to file %s\n", path);
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
    for (i = 0; i < ((num_input+1)*(num_hidden+1)); i++) {
        if(!fscanf(file, "%lf\n", &nnet->hidden_weights[i])){
            printf("Error in hidden weight reading\n");
        }
    }
    for (i = 0; i < ((num_hidden+1)*(num_output+1));i++) {
        if(!fscanf(file, "%lf\n", &nnet->output_weights[i])){
            printf("Error in output weight reading\n");
        }
    }
    for (i = 0; i < ((num_training+1)*(num_hidden+1)); i++) {
        if(!fscanf(file, "%lf\n", &nnet->hidden[i])){
            printf("Error in hidden layer reading\n");
        }
    }
    for (i = 0; i < ((num_hidden+1)*(num_output+1)); i++) {
        if(!fscanf(file, "%lf\n", &nnet->output[i])){
            printf("Error in output layer reading\n");
        }
    }
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



