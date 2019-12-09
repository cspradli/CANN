/*#include <stdlib.h>
#include <stdio.h>

double init_weights(){
    return ((double) rand())/((double) RAND_MAX);
}

int main(int argc, char const *argv[])
{
    double twoD[5][5];
    double oneD[25];
    int i = 0;
    int j = 0;
    for (i = 0; i < 5; i++){
        for (j = 0; j < 5; j++){
            double filler = init_weights();
            twoD[i][j] = filler;
            oneD[i * 5 + j] = filler;
        }
    }
    printf("[");
    for (i = 0; i < 5; i++){
        printf("\n");
        for (j = 0; j < 5; j++){
            //double filler = init_weights();
            printf("%f, ", twoD[i][j]);
            //oneD[i * 5 + j] = filler;
        }
    }
    printf("]\n");
    printf("[");
    for (i = 0; i < 25; i++){
        if ((i % 5) == 0){
            printf("\n");
        }
        printf("%f, ", oneD[i]);
    }
    printf("]\n");
    for (i = 0; i < 5; i++){
        printf("\n");
        for (j = 0; j < 5; j++){
            //double filler = init_weights();
            printf("2D: %f, ", twoD[i][j]);
            printf("1D: %f\n", oneD[i * 5 + j]);
            //oneD[i * 5 + j] = filler;
        }
    }
    printf("%f\n", twoD[4][4]);
    printf("%f\n", oneD[4 * 5 + 4]);
    return 0;
}*/
// C program to print *  
// in place of characters 
#include<stdio.h> 
#include<stdlib.h> 
int main(void){ 
    char password[55]; 
  
    printf("password:\n"); 
    int p=0; 
    do{ 
        password[p]=getc(stdin); 
        if(password[p]!='\r'){ 
            printf("*"); 
        } 
        p++; 
    }while(password[p-1]!='\r'); 
    password[p-1]='\0'; 
    printf("\nYou have entered %s as password.",password); 
    getch(); 
} 