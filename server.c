#include <stdio.h>
#include <stdlib.h>


void server_job(int argc, char **argv){
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

}