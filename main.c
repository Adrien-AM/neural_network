#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "utils.h"

// Example function to approximate
float f(float x)
{
    return (0.23 * x * x) + 3.2 * x;
}

int main(void)
{
    srand(time(NULL));

    // define all parameters
    // for data
    const size_t DATA_SIZE = 5000;
    const size_t INPUT_SIZE = 1;
    const size_t OUTPUT_SIZE = 1;

    // for model
    const float learning_rate = 1e-5;
    const float momentum_constant = 0.3;
    const float initializer_mean = 0;
    const float initializer_stddev = 0.5;
    const int use_bias = 1;
    const size_t training_epochs = 15;

    // define model shape
    size_t layers_size[] = {4, 1};
    float (*activations[])(float, int) = {&relu, &linear};

    struct neural_network *nn = create_model(2, layers_size, INPUT_SIZE, activations);
    randomize_weights(nn, initializer_mean, initializer_stddev, use_bias);

    // generate data
    float *inputs[DATA_SIZE];
    float *outputs[DATA_SIZE];
    for (size_t i = 0; i < DATA_SIZE; i++)
    {
        inputs[i] = malloc(sizeof(float) * INPUT_SIZE);
        outputs[i] = malloc(sizeof(float) * OUTPUT_SIZE);
        inputs[i][0] = (float)rand() / (float)(RAND_MAX/50) - 25;
        outputs[i][0] = f(inputs[i][0]);
    }

    // Start training !
    fit(nn, DATA_SIZE, inputs, outputs, training_epochs, learning_rate, momentum_constant);

    // Test on a fixed example
    float inp[] = {5};
    float *result = feed_forward(nn, inp);
    printf("Input : %f, result : %f\n", inp[0], result[0]);
 
 
    free(result);
    for (size_t i = 0; i < DATA_SIZE; i++) 
    {
        free(inputs[i]);
        free(outputs[i]);
    }
    free_neural_network(nn);

    return EXIT_SUCCESS;
}