#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "data_utils.h"
#include "utils.h"

int main(void)
{
    struct csv *house_data = read_csv("./data/houses/house_data_v2.csv");
    struct csv *price_house_data = extract_target_from_data("price", house_data);
    struct norm normalization =
      get_norm_parameters(house_data->data, house_data->nb_columns, house_data->nb_lines);
    normalize_inputs(house_data->data, house_data->nb_columns, house_data->nb_lines, normalization);

    // define all parameters
    // for data
    const size_t INPUT_SIZE = 6;
    const size_t OUTPUT_SIZE = 1;

    // for model
    const double learning_rate = 1e-6;
    const double momentum_constant = 0.2;
    const double initializer_mean = 0;
    const double initializer_stddev = 1;
    const int use_bias = 1;
    const size_t training_epochs = 7;
    const size_t batch_size = 1;

    // define model shape
    size_t layers_size[] = {4, 2, OUTPUT_SIZE};
    double (*activations[])(double, int) = {&relu, &relu, &linear};

    struct neural_network *nn = create_model(3, layers_size, INPUT_SIZE, activations);
    randomize_weights(nn, initializer_mean, initializer_stddev, use_bias);
    fit(nn, house_data->nb_lines, house_data->data, price_house_data->data, training_epochs, batch_size, learning_rate, momentum_constant);

    struct csv *test_data = read_csv("./data/houses/validation_data.csv");
    struct csv *price_test_data = extract_target_from_data("price", test_data);
    normalize_inputs(test_data->data, test_data->nb_columns, test_data->nb_lines, normalization);

    evaluate(nn, test_data->nb_lines, test_data->data, price_test_data->data, &mean_absolute_error, 1);

    free_neural_network(nn);
    free_csv(price_test_data);
    free_csv(test_data);
    free_csv(price_house_data);
    free_csv(house_data);

    return EXIT_SUCCESS;
}