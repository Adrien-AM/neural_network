#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data_utils.h"

#define RED "\x1b[31m"
#define BLANK "\033[0m"
#define GREEN "\e[0;32m"

void my_assert(int cond, char *message)
{
    if (!cond)
    {
        printf("%sAssertion error%s : %s\n", RED, BLANK, message);
        exit(1);
    }
}

int main(void)
{
    struct csv * csv = read_csv("test.csv");
    my_assert(csv->nb_columns == 2, "Wrong number of colors");
    my_assert(strcmp(csv->columns[0], "c1") == 0, "Wrong first column");
    my_assert(strcmp(csv->columns[1], "c2") == 0, "Wrong second column");

    my_assert(csv->nb_lines == 2, "Wrong number of lines");
    
    // Cannot test strict equality on doubles
    my_assert(fabs(csv->data[0][0] - 22.) < 1e-6, "Wrong 0-0 data value");
    my_assert(fabs(csv->data[0][1] - 3.) < 1e-6, "Wrong 0-1 data value");
    my_assert(fabs(csv->data[1][0] - 4.) < 1e-6, "Wrong 1-0 data value");
    my_assert(fabs(csv->data[1][1] - 5.3) < 1e-6, "Wrong 1-1 data value");

    printf("%sAll tests passed !%s\n", GREEN, BLANK);
    free_csv(csv);

    return EXIT_SUCCESS;
}