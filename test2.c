#include <stdio.h>
#include "nnet.h"
#include <stdlib.h>
#include <string.h>
#include "server.h"
#include "nnet_io.h"

int main(int argc, char const *argv[])
{
    data *new;
    new = get_data("./test_data", 2, 1);

    printf("%s\n", get_ln(fopen("./yee", "r")));
    printf("%d\n", get_lines("./test_data"));
    return 0;
}
