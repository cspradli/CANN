#ifndef mycurl_h
#define mycurl_h

#include <pthread.h>
#include "linked_list.h"

char* parse_and_fetch_url(char*url);


char* mycurl(char *address_to_connect_to, int port_to_connect_to, char *webpage_to_get);

struct arg {
    int i;
    char*url;
    linked_list*my_list;
};

void *worker_thread(void * targ);


#endif /* mycurl_h */
