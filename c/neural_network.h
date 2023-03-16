#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#include "utils.h"

struct neuron
{
    double* weights;
    double bias;
    double value;
    double actv_value;
    double error;
    double momentum;
};

struct neural_network
{
    struct layer** layers;
    size_t number_of_layers;
    struct loss loss;
    int use_bias;
    double gradient_clip;
};

/*
 * Allocates sequential model from given architecture.
 * @param loss : loss function (y_true, y_pred, size)
 * @param use_bias : boolean. set to 0 to ignore bias in neurons
 * @param input_size : dimension of input
 * @param number_of_layers : should correspond to the number of varargs given
 * @param gradient_clip : max value for gradient to avoid gradient explosion. Will not be used if 0
 * @param ... : layers of the model. They should be initialized, except for input_size which is
 * computed by this function.
 */
struct neural_network*
create_model(struct loss loss,
             int use_bias,
             double gradient_clip,
             size_t input_size,
             size_t number_of_layers,
             ...);

void
reset_values(struct neural_network* nn);

void
free_neural_network(struct neural_network* nn);

/*
 * Sets random weights to neural network. Weights are randomly chosen from a normal distribution of
 * mean mu and stddev sigma.
 */
void
randomize_weights(struct neural_network* nn, double mu, double sigma);

/*
 * @param nn
 * @param inputs vector to feed neural network
 *
 * @return prediction vector
 */
double*
predict(struct neural_network* nn, double inputs[], size_t nb_inputs);

/*
 * Computes back propagation after a feed forward
 * @param nn
 * @param output desired output vector
 * @param inputs used in forward prop.
 * @param learning_rate alpha, step size of gradient.
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma =
 * lower momentum.
 */
void
back_propagate(struct neural_network* nn,
               double* output,
               double inputs[],
               double learning_rate,
               double gamma);

/*
 * Fits the model to given inputs and outputs.
 *
 * @param nn
 * @param data_size number of lines in dataset
 * @param inputs matrix of train inputs
 * @param outputs matrix of train outputs
 * @param epochs number of iterations
 * @param batch_size
 * @param learning_rate step size of gradient
 * @param gamma momentum constant. If gamma = 0, doesn't have impact on computation. Lower gamma =
 * lower momentum.
 */
void
fit(struct neural_network* nn,
    size_t data_size,
    double* inputs[],
    double* outputs[],
    size_t epochs,
    size_t batch_size,
    double learning_rate,
    double gamma);

/*
 * Evaluates a trained model over a new data set.
 * @param nn
 * @param data_size number of lines in dataset
 * @param inputs
 * @param outputs expected output matrix
 * @param loss loss function
 * @param verbose display mode. Set to 2 for every output, 1 for only final loss.
 */
double
evaluate(struct neural_network* nn,
         size_t data_size,
         double* inputs[],
         double* outputs[],
         struct loss loss,
         int verbose);

// Raw dump
void
save_nn(struct neural_network* nn, char* filename);
void
read_nn(struct neural_network* nn, char* filename);

#endif // __NEURAL_NETWORK_H__
