#include "layer.h"
#include "neural_network.h"

void
instanciate_neurons(struct layer* layer)
{
    // instanciate array of neurons
    layer->neurons = (struct neuron**)malloc(sizeof(struct neuron*) * layer->size);
    for (size_t n = 0; n < layer->size; n++) {
        struct neuron* neuron = (struct neuron*)malloc(sizeof(struct neuron));

        // First layer receives input
        neuron->weights = (double*)calloc(layer->input_size, sizeof(double));
        neuron->bias = 0;
        neuron->value = 0;
        neuron->actv_value = 0;
        neuron->error = 0;
        neuron->momentum = 0;

        layer->neurons[n] = neuron;
    }
}

void
dense_forward(struct layer* layer, struct layer* input_layer)
{

    for (size_t n = 0; n < layer->size; n++) {
        for (size_t i = 0; i < layer->input_size; i++) {
            layer->neurons[n]->value +=
              input_layer->neurons[i]->actv_value * layer->neurons[n]->weights[i]; // xi * wi
        }
        layer->neurons[n]->actv_value = layer->activation(layer->neurons[n]->value, 0);
    }
}

void
dense_backprop(struct layer* layer, struct layer* input_layer)
{
    for (size_t n = 0; n < layer->size; n++) {
        struct neuron* neuron = layer->neurons[n];

        // here error = dactv
        // compute derivative on current neuron error
        neuron->error *= layer->activation(neuron->actv_value, 1);
        // here error = dz

        if (input_layer != NULL) {

            // propagate to previous layer, weighted
            for (size_t i = 0; i < layer->input_size; i++) {
                input_layer->neurons[i]->error += neuron->error * neuron->weights[i];
            }
        }
    }
}

struct layer*
dense_layer(size_t number_of_neurons, double (*activation)(double, int))
{
    struct layer* layer = malloc(sizeof(struct layer));
    layer->size = number_of_neurons;
    layer->activation = activation;
    layer->forward = &dense_forward;
    layer->backprop = &dense_backprop;
    layer->input_size = 0; // unknown yet

    return layer;
}

void
softmax_forward(struct layer* layer, struct layer* input_layer)
{
    // Softmax layer should be same size as input layer
    if (layer->size != input_layer->size) {
        printf("Softmax layer should have same size as previous layer.\n");
        exit(0);
    }
    double denominator = 0;
    for (size_t i = 0; i < input_layer->size; i++) {
        layer->neurons[i]->actv_value = exp(input_layer->neurons[i]->actv_value);
        denominator += layer->neurons[i]->actv_value;
    }

    for (size_t i = 0; i < input_layer->size; i++) {
        layer->neurons[i]->actv_value /= denominator;
    }
}

// https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
void
softmax_backprop(struct layer* layer, struct layer* input_layer)
{
    for (size_t j = 0; j < input_layer->size; j++) {
        for (size_t i = 0; i < layer->size; i++) {
            input_layer->neurons[j]->error += layer->neurons[i]->error * 
              layer->neurons[i]->actv_value * ((i == j) - layer->neurons[j]->actv_value);
        }
    }
}

struct layer*
softmax_layer(size_t number_of_classes)
{
    struct layer* layer = malloc(sizeof(struct layer));
    layer->activation = linear;
    layer->size = number_of_classes;
    layer->input_size = 0; // unknown
    layer->forward = &softmax_forward;
    layer->backprop = &softmax_backprop;

    return layer;
}