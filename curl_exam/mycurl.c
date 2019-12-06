#include <sys/socket.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netdb.h>
#include <string.h>
#include <arpa/inet.h>
#include "mycurl.h"

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

void *worker_thread(void * targ){
    struct arg* arg = (struct arg*)targ;
    char *output = parse_and_fetch_url(arg->url);
    pthread_mutex_lock(&mutex1);
    linked_list_add(arg->my_list, output);
    pthread_mutex_unlock(&mutex1);
    free(output);
    pthread_exit(0);
}

char* parse_and_fetch_url(char*url){
    // Adapted from: https://stackoverflow.com/questions/726122/best-ways-of-parsing-a-url-using-c
    char host[256];
    int port = 80;
    char uri[512] = "";
    int succ_parsing = 0;

    if (sscanf(url, "http://%99[^:]:%i/%199[^\n]", host, &port, uri) == 3) { succ_parsing = 1;}
    else if (sscanf(url, "http://%99[^/]/%199[^\n]", host, uri) == 2) { succ_parsing = 1;}
    else if (sscanf(url, "http://%99[^:]:%i[^\n]", host, &port) == 2) { succ_parsing = 1;}
    else if (sscanf(url, "http://%99[^\n]", host) == 1) { succ_parsing = 1;}

    if(!succ_parsing){
        printf("Error parsing URL; %s\n",url);
        exit(0);
    }
    char uri2[512] = "/";
    strcat(uri2, uri);

    //printf("host = \"%s\"\n", host);
    //printf("port = \"%d\"\n", port);
    //printf("uri = \"%s\"\n", uri2);
    
    return mycurl(host,port,uri2);
}

char* mycurl(char *address_to_connect_to, int port_to_connect_to, char *webpage_to_get){
    //char *address_to_connect_to = "www.cs.unca.edu";
    //int port_to_connect_to = 80;
    //char *webpage_to_get = "/~drawert/cs335/";

    // These two must be declared globally
    struct addrinfo hints;
    struct addrinfo *infoptr;

    // Create socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0){
        perror("Error creating socket: ");
        exit(1);
    }

    // DNS lookup of IP address
    hints.ai_family = AF_INET;

    if( getaddrinfo( address_to_connect_to, NULL, &hints, &infoptr) != 0 ){
        perror("Error looking up DNS address: ");
        exit(1);
    }

    char ip_address[256];
    struct addrinfo *p;

    int connect_successful = 0;

    for( p = infoptr ; p != NULL ; p = p->ai_next){
        getnameinfo(p->ai_addr, p->ai_addrlen, ip_address, sizeof(ip_address), NULL, 0 , NI_NUMERICHOST);

        struct sockaddr_in serv_addr;
        memset(&serv_addr, '0', sizeof(serv_addr));  //zero out everything
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port_to_connect_to);

        if( inet_pton(AF_INET, ip_address, &serv_addr.sin_addr) <= 0){
            perror("Address translation error: ");
            exit(1);
        }
        printf("Trying to connect() to %s:%i\n", ip_address, port_to_connect_to);
        if( connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0){
            perror("Connection error, could not connect: ");
            continue; // Try the next IP address in the list
        }else{
            connect_successful = 1;
            break;
        }
    }
    freeaddrinfo(infoptr);  // clean up 

    if( ! connect_successful){
        printf("Not able to connect\n");
        exit(1);
    }

    // Send HTTP Request
    char request_buffer[1000];
    request_buffer[0] = '\0';

    strcat(request_buffer, "GET ");
    strcat(request_buffer, webpage_to_get);
    strcat(request_buffer, " HTTP/1.0\r\n");

    strcat(request_buffer, "Host: ");
    strcat(request_buffer, address_to_connect_to);
    strcat(request_buffer, "\r\n");

    strcat(request_buffer, "\r\n");  // Must end HTTP with blank line

    send(sockfd, request_buffer, strlen(request_buffer), 0);

    // Read HTTP Response
    printf("reading HTTP response\n");
    int buffer_size = 1024;
    char *response_buffer = (char *) malloc(sizeof(char)*buffer_size);
    int bytes_read=0;
    int bytes_read_now=0;
    while( (bytes_read_now = read(sockfd, &response_buffer[bytes_read], 1024) ) > 0){
        bytes_read += bytes_read_now;
        if(bytes_read >= buffer_size-1){
            //realloc response buffer
            buffer_size += 1024;
            response_buffer = (char *) realloc(response_buffer,sizeof(char)*buffer_size);
        }
    }
    //printf("%s", request_buffer);
    //printf("%s\n\n", response_buffer);


    // Find response body
    char* response_body_ptr = strstr(response_buffer, "\r\n\r\n");
    if(response_body_ptr == NULL){
        printf("Error, no response body\n");
        exit(0);
    }
    char*body = malloc(sizeof(char)*(1+strlen(response_body_ptr)));
    strcpy(body,response_body_ptr+4);
    free(response_buffer);



    return body;
}

