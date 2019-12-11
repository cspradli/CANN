#include <stdio.h>
#include <stdlib.h>

int create_file(const char *path);

int main(int argc, char const *argv[])
{
    if (argc == 2){
        if(!create_file(argv[1])){
            printf("Created file %s\n", argv[1]);
        }
    }
    return 0;
}


int create_file(const char *path){
    if(!fopen(path, "a+")){
        printf("Error in creating file\n");
        return -1;

    } else
    {
        return 0;
    }
    

}