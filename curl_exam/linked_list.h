#ifndef linked_list_h
#define linked_list_h

// Create data structure for a node of the list

struct node_t {
    char * data;
    struct node_t * next;
};
typedef struct node_t node;

// Data structre for the linked list type

typedef struct {
    node*head;
    int count;
} linked_list;

// Functions to manipulate the linked list

//constructor
linked_list* create_linked_list();

// destructor
void destroy_linked_list( linked_list* ll );

// add a new node to the end of the linked list
void linked_list_add( linked_list* ll, char* data_in);

// Delete a node from the linked list
void linked_list_delete( linked_list* ll, node* to_delete);

// search for a node by it's data field
node* linked_list_search( linked_list* ll, char* search_string );

// get node by index
node* linked_list_get( linked_list* ll, int index);


#endif /* linked_list_h*/
