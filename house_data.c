#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "data_utils.h"

int main(void)
{
    struct csv *house_data = read_csv("./kc_house_data.csv");
    struct csv *price_house_data = extract_target_from_data("price", house_data);

    free_csv(price_house_data);
    free_csv(house_data);

    return EXIT_SUCCESS;
}