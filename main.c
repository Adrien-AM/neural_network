#include <stdio.h>
#include <stdlib.h>

#include "data_utils.h"
#include "layer.h"
#include "neural_network.h"
#include "utils.h"

// Example function to approximate
double
f(double* x)
{
    return (0.23 * *x * *x) + 3.2 * *x;
}

double
f2(double* x)
{
    return (-123.78 * x[0]) + (648.25 * x[1]);
}

int
main(void)
{
    srand(time(NULL));

    // define all parameters
    // for data
    const size_t DATA_SIZE = 20000;
    const size_t INPUT_SIZE = 2;
    const size_t OUTPUT_SIZE = 1;
    const size_t TEST_SIZE = 500;

    // for model
    const double learning_rate = 1e-7;
    const double momentum_constant = 1e-3;
    const double initializer_mean = 0;
    const double initializer_stddev = 1;
    const int use_bias = 0;
    const size_t training_epochs = 20;
    const size_t batch_size = 1;
    const double gradient_clip = 5;

    // define model shape
    struct neural_network* nn = create_model(
        mean_squared_error, use_bias, gradient_clip, INPUT_SIZE, 4,
        dense_layer(16, &relu),
        dropout_layer(16, 0.1),
        dense_layer(8, &relu),
        dense_layer(OUTPUT_SIZE, &linear)
    );

    randomize_weights(nn, initializer_mean, initializer_stddev);

    // generate data
    double* inputs[DATA_SIZE];
    double* outputs[DATA_SIZE];

    generate_data_inputs(DATA_SIZE, INPUT_SIZE, inputs, -10, 10);
    generate_data_outputs(DATA_SIZE, OUTPUT_SIZE, inputs, outputs, &f2);

    struct norm normalization = get_norm_parameters(inputs, INPUT_SIZE, DATA_SIZE);
    normalize_inputs(inputs, INPUT_SIZE, DATA_SIZE, normalization);

    // Start training !
    fit(nn,
        DATA_SIZE,
        inputs,
        outputs,
        training_epochs,
        batch_size,
        learning_rate,
        momentum_constant);

    for (size_t i = 0; i < DATA_SIZE; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }

    double* test_inputs[TEST_SIZE];
    double* test_outputs[TEST_SIZE];

    generate_data_inputs(TEST_SIZE, INPUT_SIZE, test_inputs, -10, 10);
    generate_data_outputs(TEST_SIZE, OUTPUT_SIZE, test_inputs, test_outputs, &f2);
    normalize_inputs(test_inputs, INPUT_SIZE, TEST_SIZE, normalization);

    evaluate(nn, TEST_SIZE, test_inputs, test_outputs, mean_squared_error, 1);

    for (size_t i = 0; i < TEST_SIZE; i++) {
        free(test_inputs[i]);
        free(test_outputs[i]);
    }

    free_neural_network(nn);

    return EXIT_SUCCESS;
}