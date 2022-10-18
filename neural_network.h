#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/*
 * Allocates empty model with weights and biases set to 0.
 */
struct neural_network *create_model(size_t number_of_layers, size_t layers_size[],
                                    size_t input_size, float (**activation)(float, int));

void free_neural_network(struct neural_network *nn);

/*
 * Sets random weights to neural network. Weights are randomly chosen from a normal distribution of mean mu and stddev sigma.
 */
void randomize_weights(struct neural_network *nn, float mu, float sigma, int use_bias);

/*
 * @param nn
 * @param inputs vector to feed neural network
 * 
 * @return prediction vector
 */
float *feed_forward(struct neural_network *nn, float inputs[]);


/*
 * Computes back propagation after a feed forward
 * @param nn
 * @param output desired output vector
 * @param inputs used in forward prop.
 * @param learning_rate alpha, step size of gradient.
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma = lower momentum.
 */
void back_propagate(struct neural_network *nn, float *output, float inputs[], float learning_rate, float gamma);

/* 
 * Fits the model to given inputs and outputs.
 *
 * @param nn
 * @param inputs matrix of train inputs
 * @param outputs matrix of train outputs
 * @param epochs number of iterations
 * @param learning_rate step size of gradient
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma = lower momentum.
 */
void fit(struct neural_network *nn, size_t data_size, float *inputs[], float *outputs[], size_t epochs, float learning_rate, float gamma);

#endif // __NEURAL_NETWORK_H__
