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
                                    size_t input_size, double (**activation)(double, int));

void free_neural_network(struct neural_network *nn);

/*
 * Sets random weights to neural network. Weights are randomly chosen from a normal distribution of mean mu and stddev sigma.
 */
void randomize_weights(struct neural_network *nn, double mu, double sigma, int use_bias);

/*
 * @param nn
 * @param inputs vector to feed neural network
 *
 * @return prediction vector
 */
double *predict(struct neural_network *nn, double inputs[], size_t nb_inputs);

/*
 * Computes back propagation after a feed forward
 * @param nn
 * @param output desired output vector
 * @param inputs used in forward prop.
 * @param learning_rate alpha, step size of gradient.
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma = lower momentum.
 */
void back_propagate(struct neural_network *nn, double *output,
                    double inputs[], double learning_rate, double gamma);

/*
 * Fits the model to given inputs and outputs.
 *
 * @param nn
 * @param data_size number of lines in dataset
 * @param inputs matrix of train inputs
 * @param outputs matrix of train outputs
 * @param epochs number of iterations
 * @param learning_rate step size of gradient
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma = lower momentum.
 */
void fit(struct neural_network *nn, size_t data_size, double *inputs[],
         double *outputs[], size_t epochs, size_t batch_size, double learning_rate, double gamma);

/*
 * Evaluates a trained model over a new data set.
 * @param nn
 * @param data_size number of lines in dataset
 * @param inputs
 * @param outputs expected output matrix
 * @param loss loss function
 * @param verbose display mode. Set to true for prints.
 */
double evaluate(struct neural_network *nn, size_t data_size,
               double *inputs[], double *outputs[], double (*loss)(double *, double *, size_t), int verbose);

#endif // __NEURAL_NETWORK_H__
