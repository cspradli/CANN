#include <stdio.h>
#include <stdlib.h>
#include "mycurl.h"


int main(int argc, char**argv){
    pthread_t tid[10];
    char* urls[10] = {
        "127.0.0.1:9114/cgi-bin/my-nnet",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check2.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check3.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check4.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check5.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check6.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check7.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check8.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check9.php",
        "http://www.cs.unca.edu/~drawert/cs335/exam2/check10.php"
    };
    int i;

    linked_list* my_list = create_linked_list();
    
    struct arg targ[10];
    for(i=0; i< 10; i++){
        printf("%s\n", urls[i]);
        targ[i].i = i;
        targ[i].url = urls[i];
        targ[i].my_list = my_list;
        pthread_create(&tid[i], NULL, worker_thread, (void *)&targ[i]);
    }

    for (i=0; i < 10; i++){
        pthread_join(tid[i], NULL);
    }

    // print out list
    node* n;
    for(i=0; i < my_list->count; i++){
        n = linked_list_get(my_list, i);
        printf("[%i] %s\n", i, n->data);
    }

    // clean up
    destroy_linked_list( my_list );
    return 0;
}





