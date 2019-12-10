#include "server.h"

int get_input(int argc, char const *argv[]){

    if (argc < 3) printf("USAGE: './my-nnet y/n username@host.edu' (y/n for send to server option)\n");
    if (!strcmp(argv[1], "y")){
        const char *input = argv[2];
        printf("GETTING INPUT %s \n", input);
        parse_input(input);
        return 0;
    } else if (!strcmp(argv[1], "n")){
        printf("Running locally\n");
        return 1;
    } else {
        printf("Inproper usage: need either 'y' or 'n'\n");
        return -1;
    }
    fflush(stdin);
    fflush(stdout);
}

int parse_input(const char *input){
    int ret = 0;
    char username[250];
    char host[250];
    char output[250];
        if (sscanf(input, "%99[^@]@%99[^\n]", username, host) == 2){
            printf("Succesful username and host capture\n");
            ret = 1;
        } else {
            printf("Unsuccesful\n");
            printf("Need correct username and host to run on specified server\n");
            ret = 0;
        }
    printf("USERNAME: %s\n", username);
    printf("HOST:%s\n",host);
    strcat(output, username);
    strcat(output, "@");
    strcat(output, host);
    printf("SANITY CHECK: %s\n", output);
    fflush(stdin);
    fflush(stdout);
    if (ret == 1){
        if (!check_input()) exec_all(output);
        return ret;
    } else {
        printf("ERROR in parsing\n");
        return 0;
    }

}


int check_input(){
    char check[8];
    char *output;
    printf("\nNext steps include:\n(1) Copying this executable to the server specified (using wget)\n(2) Running this executable through SSH tunnel\nNote: Password will never be shared to this program\nThe only time password should be asked is from server specified's shell\n\n");
    printf("Proceed? (yes/no) ");
    output = fgets(check, 8, stdin);
    if (strcmp(check, "yes")){
        printf("good: %s", output);
        return 0;
    } else {
        printf("bad: %s", output);
        return -1;
    }
}

void exec_wget(char *input, int check){
    printf("Executing wget to transfer executable to %s\n", input);
    char ssh_wget[256];
    check = 0;
    strcat(ssh_wget, "ssh ");
    strcat(ssh_wget, input);
    strcat(ssh_wget, " 'wget http://arden.cs.unca.edu/~cspradli/my-nnet'");
    printf("SSH COMMAND 0: %s\nHanding over control to sh\n", ssh_wget);
    fflush(stdin);
    fflush(stdout);
    if (system(ssh_wget)){
        printf("ERROR IN SSH WGET\n");
        check = 0;
    } else {
        check = 1;
        sleep(2);
        exec_job(input, check);
    }
}

void exec_job(char *input, int check){
    printf("Executing job on server %s\n", input);
    char ssh_run[256];
    check = 0;
    strcat(ssh_run, "ssh ");
    strcat(ssh_run, input);
    strcat(ssh_run, " './my-nnet n' > output.txt");
    printf("SSH COMMAND 1: %s\nHanding over control to sh\n", ssh_run);
    fflush(stdin);
    fflush(stdout);
    if (system(ssh_run)){
        printf("ERROR IN SSH RUN\n");
        check = 0;
    } else {
        check = 1;
        printf("check %d\n", check);
    }
}

void exec_all(char *input){
    printf("Executing job on server %s\n", input);
    char ssh_run[512];
    strcat(ssh_run, "ssh ");
    strcat(ssh_run, input);
    strcat(ssh_run, " 'wget http://arden.cs.unca.edu/~cspradli/my-nnet; sleep 1; ./my-nnet n;' > output.txt");
    fflush(stdin);
    fflush(stdout);
    if (system(ssh_run)){
        printf("ERROR IN SSH\n");
    } else {
        printf("Succesful server job: check output.txt for output\n");
    }
}



