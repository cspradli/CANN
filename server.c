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
    printf("USRN:%s\n", username);
    printf("HST:%s\n",host);
    strcat(output, username);
    strcat(output, "@");
    strcat(output, host);
    printf("OUTPUT %s\n", output);
    if (ret == 1){
        exec_wget(output, ret);
        return ret;
    } else {
        printf("ERROR in parsing\n");
        return 0;
    }

}

void exec_wget(char *input, int check){
    printf("Executing wget to transfer executable to %s\n", input);
    char ssh_wget[256];
    check = 0;
    strcat(ssh_wget, "ssh ");
    strcat(ssh_wget, input);
    strcat(ssh_wget, " 'wget http://arden.cs.unca.edu/~cspradli/my-nnet'\0");
    printf("SSH COMMAND 0: %s\n", ssh_wget);
    if (!system(ssh_wget)){
        printf("ERROR IN SSH\n");
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
    strcat(ssh_run, " './my-nnet n' > output.txt\0");
    printf("SSH COMMAND 1: %s\n", ssh_run);
    if (!system(ssh_run)){
        printf("ERROR IN SSH\n");
        check = 0;
    } else {
        check = 1;
        printf("check %d\n", check);
    }
}


/*void server_job(int argc, char **argv){
    if (argc < 2){
        printf("USAGE: './my-nnet (send to server)y/n'\n");
    }
    if (strcmp(argv[1], "y") == 0){
        printf("SENDING JOB TO SERVER\n");
        if (fork() == 0){ //spin off child to run server commands
            printf("get command to ssh\n");
            if(!system("ssh cspradli@montreat.cs.unca.edu 'wget http://arden.cs.unca.edu/~cspradli/my-nnet'"))printf("ERROR: 505\n");
            if(!system("ssh cspradli@montreat.cs.unca.edu './my-nnet n' > output.txt")) printf("ERROR: 505\n");
        }
    } else {
    }

}*/