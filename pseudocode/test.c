#include <stdlib.h>
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
            twoD[j][i] = filler;
            oneD[j + 5 * i] = filler;
        }
    }
    printf("[");
    for (i = 0; i < 5; i++){
        printf("\n");
        for (j = 0; j < 5; j++){
            //double filler = init_weights();
            printf("%f, ", twoD[j][i]);
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
            printf("%d, %d, 2D: %f, ", i, j, twoD[j][i]);
            printf("%d, %d, 1D: %f\n", i, j, oneD[j + 5 * i]);
            //oneD[i * 5 + j] = filler;
        }
    }
    printf("%f\n", twoD[3][1]);
    printf("%f\n", oneD[3 + 5 * 1]);
    return 0;
}
// C program to print *  
// in place of characters 
/*#include<stdio.h> 
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
} */